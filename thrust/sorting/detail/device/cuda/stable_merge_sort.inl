/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


/*! \file stable_merge_sort_dev.inl
 *  \brief Inline file for stable_merge_sort_dev.h.
 *  \note This algorithm is based on the one described
 *        in "Designing Efficient Sorting Algorithms for
 *        Manycore GPUs", by Satish, Harris, and Garland.
 *        Nadatur Satish is the original author of this
 *        implementation.
 */

// do not attempt to compile this file with any other compiler
#ifdef __CUDACC__

#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/utility.h>

#include <thrust/sorting/detail/device/cuda/block/merging_sort.h>

namespace thrust
{

namespace sorting
{

namespace detail
{

namespace device
{

namespace cuda
{

namespace merge_sort_dev_namespace
{

// define our own min() rather than #include <thrust/extrema.h>
template<typename T>
  __host__ __device__
  T min(const T &lhs, const T &rhs)
{
  return lhs < rhs ? lhs : rhs;
} // end min()

// compute the log base-2 of an integer at compile time
template <unsigned int N, unsigned int Cur = 0>
struct lg
{
  static const unsigned int result = lg<(N >> 1),Cur+1>::result;
};

template <unsigned int Cur>
struct lg<1,Cur>
{
  static const unsigned int result = Cur;
};

template<typename Key, typename Value>
  class BLOCK_SIZE
{
  private:
    static const unsigned int sizeof_larger_type = (sizeof(Key) > sizeof(Value) ? sizeof(Key) : sizeof(Value));

    // our block size candidate is simply 2K over the sum of sizes
    static const unsigned int candidate = 2048 / (sizeof(Key) + sizeof(Value));

    // round one_k_over_size down to the nearest power of two
    static const unsigned int lg_candidate = lg<candidate>::result;

    // exponentiate that result, which rounds down to the nearest power of two
    static const unsigned int final_candidate = 1<<lg_candidate;

  public:
    static const unsigned int result = 128 < final_candidate ? 128 : final_candidate;
};

template<typename Key, typename Value>
  class LOG_BLOCK_SIZE
{
  private:
    static const unsigned int block_size = BLOCK_SIZE<Key,Value>::result;

  public:
    static const unsigned int result = lg<block_size>::result;
};

static const unsigned int WARP_SIZE = 32;

inline unsigned int max_grid_size(const unsigned int block_size)
{
  return 3 * thrust::experimental::arch::max_active_threads() / block_size;
} // end max_grid_size()

// Base case for the merge algorithm: merges data where tile_size <= BLOCK_SIZE. 
// Works by loading two or more tiles into shared memory and doing a binary search.
template<unsigned int BLOCK_SIZE,
         typename Tkey,
         typename Tvalue,
         typename StrictWeakOrdering>
__global__ void merge_smalltiles_binarysearch(const Tkey * d_srcDatakey, const Tvalue * d_srcDatavalue,
                                              const unsigned int n,
                                              const unsigned int index_of_last_block,
                                              const unsigned int index_of_last_tile_in_last_block,
                                              const unsigned int size_of_last_tile,
                                              Tkey * d_resultDatakey, Tvalue *d_resultDatavalue,
                                              const unsigned int log_tile_size,
                                              StrictWeakOrdering comp)
{
  // Assumption: tile_size is a power of 2.
  
  // load (2*BLOCK_SIZE) elements into shared memory. These (2*BLOCK_SIZE) elements belong to (2*BLOCK_SIZE)/tile_size different tiles.
  // XXX workaround no constructors in shared array problem
  __shared__ unsigned char key_workaround[(2*BLOCK_SIZE) * sizeof(Tkey)];
  Tkey *key = reinterpret_cast<Tkey*>(key_workaround);

  __shared__ unsigned char outkey_workaround[(2*BLOCK_SIZE) * sizeof(Tkey)];
  Tkey *outkey = reinterpret_cast<Tkey*>(outkey_workaround);

  __shared__ unsigned char outvalue_workaround[(2*BLOCK_SIZE) * sizeof(Tvalue)];
  Tvalue *outvalue = reinterpret_cast<Tvalue*>(outvalue_workaround);

  const unsigned int grid_size = gridDim.x * blockDim.x;

  // this will store the final rank of our key
  unsigned int rank;

  unsigned int block_idx = blockIdx.x;
  
  // the global index of this thread
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

  for(;
      block_idx <= index_of_last_block;
      block_idx += gridDim.x, i += grid_size)
  {
    // figure out if this thread should idle
    unsigned int thread_not_idle = i < n;
    Tkey my_key;
    
    // copy over inputs to shared memory
    if(thread_not_idle)
    {
      key[threadIdx.x] = my_key = d_srcDatakey[i];
    } // end if
    
    // the tile to which the element belongs
    unsigned int tile_index = threadIdx.x>>log_tile_size;

    // figure out the index and size of the other tile
    unsigned int other_tile_index = tile_index^1;
    unsigned int other_tile_size = (1<<log_tile_size);

    // if the other tile is the final tile, it is potentially
    // smaller than the rest
    if(block_idx == index_of_last_block
       && other_tile_index == index_of_last_tile_in_last_block)
    {
      other_tile_size = size_of_last_tile;
    } // end if
    
    // figure out where the other tile begins in shared memory
    Tkey * other = key + (other_tile_index<<log_tile_size);
    
    // do a binary search on the other tile, and find the rank of the element in the other tile.
    unsigned int start, end, cur;
    start = 0; end = other_tile_size;
    
    // binary search: at the end of this loop, start contains
    // the rank of key[threadIdx.x] in the merged sequence.

    __syncthreads();
    if(thread_not_idle)
    {
      while(start < end)
      {
        cur = (start + end)>>1;

        // these are uncoalesced accesses: use shared memory to mitigate cost.
        if((comp(other[cur], my_key))
           || (!comp(my_key, other[cur]) && (tile_index&0x1)))
        {
          start = cur + 1;
        } // end if
        else
        {
          end = cur;
        } // end else
      } // end while

      // to compute the rank of my element in the merged sequence
      // add the rank of the element in the other tile
      // plus the rank of the element in this tile
      rank = start + ((tile_index&0x1)?(threadIdx.x -(1<<log_tile_size)):(threadIdx.x));
    } // end if

    if(thread_not_idle)
    {
      // these are scatters: use shared memory to reduce cost.
      outkey[rank] = my_key;
      outvalue[rank] = d_srcDatavalue[i];
    } // end if
    __syncthreads();
    
    if(thread_not_idle)
    {
      // coalesced writes to global memory
      d_resultDatakey[i] = outkey[threadIdx.x];
      d_resultDatavalue[i] = outvalue[threadIdx.x];
    } // end if
    __syncthreads();
  } // end for
} // end merge_smalltiles_binarysearch()

template<unsigned int BLOCK_SIZE,
         typename KeyType,
         typename ValueType,
         typename StrictWeakOrdering>
  __global__ void stable_odd_even_block_sort_kernel(KeyType *keys,
                                                    ValueType *data,
                                                    StrictWeakOrdering comp,
                                                    const unsigned int n)
{
  // XXX workaround no constructors on device arrays
  __shared__ unsigned char s_keys_workaround[BLOCK_SIZE * sizeof(KeyType)];
  KeyType *s_keys = reinterpret_cast<KeyType*>(s_keys_workaround);

  __shared__ unsigned char s_data_workaround[BLOCK_SIZE * sizeof(ValueType)];
  ValueType *s_data = reinterpret_cast<ValueType*>(s_data_workaround);

  const unsigned int grid_size = gridDim.x * blockDim.x;

  // block_offset records the global index of this block's 0th thread
  unsigned int block_offset = blockIdx.x * BLOCK_SIZE;
  unsigned int i = threadIdx.x + block_offset;

  for(;
      block_offset < n;
      block_offset += grid_size, i += grid_size)
  {
    __syncthreads();
    // copy input to shared
    if(i < n)
    {
      s_keys[threadIdx.x] = keys[i];
      s_data[threadIdx.x] = data[i];
    } // end if
    __syncthreads();

    // this block could be partially full
    unsigned int length = BLOCK_SIZE;
    if(block_offset + BLOCK_SIZE > n)
    {
      length = n - block_offset;
    } // end if

    // run merge_sort over the block
    block::merging_sort(s_keys, s_data, length, comp);

    // write result
    if(i < n)
    {
      keys[i] = s_keys[threadIdx.x];
      data[i] = s_data[threadIdx.x];
    } // end if
  } // end for i
} // end stable_odd_even_block_sort_kernel()

// Extract the splitters: every blockDim.x'th element of src is a splitter
// Input: src, src_size
// Output: splitters: the splitters
//         splitters_pos: Index of splitter in src
template<typename ValueType>
  __global__ void extract_splitters(ValueType *src,
                                    unsigned int src_size,
                                    ValueType *splitters,
                                    unsigned int *splitters_pos)
{
  const unsigned int grid_size = gridDim.x * blockDim.x;

  unsigned int splitter_idx = blockIdx.x;
  unsigned int src_idx = splitter_idx * blockDim.x;
  for(;
      src_idx < src_size;
      src_idx += grid_size, splitter_idx += gridDim.x)
  {
    if(threadIdx.x == 0)
    {
      splitters[splitter_idx] = src[src_idx]; 
      splitters_pos[splitter_idx] = src_idx;
    } // end if
  } // end while
} // end extract_splitters()

///////////////////// Find the rank of each extracted element in both arrays ////////////////////////////////////////
///////////////////// This breaks up the array into independent segments to merge ////////////////////////////////////////
// Inputs: d_splitters, d_splittes_pos: the merged array of splitters with corresponding positions.
//		   d_srcData: input data, datasize: number of entries in d_srcData
//		   N_SPLITTERS the number of splitters, log_blocksize: log of the size of each block of sorted data
//		   log_num_merged_splitters_per_block = log of the number of merged splitters. ( = log_blocksize - 7). 
// Output: d_rank1, d_rank2: ranks of each splitter in d_splitters in the block to which it belongs
//		   (say i) and its corresponding block (block i+1).
template<unsigned int BLOCK_SIZE,
         unsigned int LOG_BLOCK_SIZE,
         typename KeyType,
         typename ValueType,
         typename StrictWeakOrdering>
  __global__ void find_splitter_ranks(KeyType *d_splitters,     unsigned int *d_splitters_pos, 
                                      unsigned int *d_rank1,  unsigned int *d_rank2, 
                                      ValueType *d_srcData,   unsigned int datasize, 
                                      unsigned int N_SPLITTERS, unsigned int log_blocksize, 
                                      unsigned int log_num_merged_splitters_per_block,
                                      StrictWeakOrdering comp)
{
  KeyType inp;
  unsigned int inp_pos;
  int start, end, cur;

  const unsigned int grid_size = gridDim.x * blockDim.x;

  unsigned int block_offset = blockIdx.x * blockDim.x;
  unsigned int i = threadIdx.x + block_offset;
  
  for(;
      block_offset < N_SPLITTERS;
      block_offset += grid_size, i += grid_size)
  {
    if(i < N_SPLITTERS)
    { 
      inp = d_splitters[i];
      inp_pos = d_splitters_pos[i];
      
      // the (odd, even) block pair to which the splitter belongs. Each i corresponds to a splitter.
      unsigned int oddeven_blockid = i>>log_num_merged_splitters_per_block;
      
      // the local index of the splitter within the (odd, even) block pair.
      unsigned int local_i  = i- ((oddeven_blockid)<<log_num_merged_splitters_per_block);
      
      // the block to which the splitter belongs.
      unsigned int listno = (inp_pos >> log_blocksize);
      
      // the "other" block which which block listno must be merged.
      unsigned int otherlist = listno^1;
      KeyType *other = d_srcData + (1<<log_blocksize)*otherlist;
      
      // the size of the other block can be less than blocksize if the it is the last block.
      unsigned int othersize = min<unsigned int>(1 << log_blocksize, datasize - (otherlist<<log_blocksize));
      
      // We want to compute the ranks of element inp in d_srcData1 and d_srcData2
      // for instance, if inp belongs to d_srcData1, then 
      // (1) the rank in d_srcData1 is simply given by its inp_pos
      // (2) to find the rank in d_srcData2, we first find the block in d_srcData2 where inp appears.
      //     We do this by noting that we have already merged/sorted d_splitters, and thus the rank
      //     of inp in the elements of d_srcData2 that are present in d_splitters is given by 
      //        position of inp in d_splitters - rank of inp in elements of d_srcData1 in d_splitters
      //        = i - inp_pos
      //     This also gives us the block of d_srcData2 that inp belongs in, since we have one
      //     element in d_splitters per block of d_srcData2.
      
      //     We now perform a binary search over this block of d_srcData2 to find the rank of inp in d_srcData2.
      //     start and end are the start and end indices of this block in d_srcData2, forming the bounds of the binary search.
      //     Note that this binary search is in global memory with uncoalesced loads. However, we only find the ranks 
      //     of a small set of elements, one per splitter: thus it is not the performance bottleneck.
      if(!(listno&0x1))
      { 
        d_rank1[i] = inp_pos + 1 - (1<<log_blocksize)*listno; 
        end = (( local_i - ((d_rank1[i] - 1)>>LOG_BLOCK_SIZE)) <<LOG_BLOCK_SIZE ) - 1; start = end - (BLOCK_SIZE-1);
        if(end < 0) start = end = 0;
        if(end >= othersize) end = othersize - 1;
        if(start > othersize) start = othersize;
      } // end if
      else
      { 
        d_rank2[i] = inp_pos + 1 - (1<<log_blocksize)*listno;
        end = (( local_i - ((d_rank2[i] - 1)>>LOG_BLOCK_SIZE)) <<LOG_BLOCK_SIZE ) - 1; start = end - (BLOCK_SIZE-1);
        if(end < 0) start = end = 0;
        if(end >= othersize) end = othersize - 1;
        if(start > othersize) start = othersize;
      } // end else
      
      // we have the start and end indices for the binary search in the "other" array
      // do a binary search. Break ties by letting elements of array1 before those of array2 
      while(start <= end)
      {
        cur = (start + end)>>1;

        // XXX eliminate the need for two comparisons here and ensure the sort is still stable
        if((comp(other[cur], inp))
           || (!comp(inp, other[cur]) && (listno&0x1)))
        {
          start = cur + 1;
        } // end if
        else
        {
          end = cur - 1;
        } // end else
      } // end while

      if(!(listno&0x1))
      {
        d_rank2[i] = start;	
      } // end if
      else
      {
        d_rank1[i] = start;	
      } // end else
    } // end if
  } // end for
} // end find_splitter_ranks()

///////////////// Copy over first merged splitter of each odd-even block pair to the output array //////////////////
template<unsigned int LOG_BLOCK_SIZE, typename KeyType, typename ValueType>
  __global__ void copy_first_splitters(KeyType * srcdatakey, ValueType * srcdatavalue, unsigned int * d_splitters_pos, 
                                       KeyType *resultdatakey, ValueType *resultdatavalue, unsigned int log_num_merged_splitters_per_block,
                                       const unsigned int num_tile_pairs)
{
  for(unsigned int block_idx = blockIdx.x;
      block_idx < num_tile_pairs;
      block_idx += gridDim.x)
  {
    unsigned int splitter_idx = block_idx << (log_num_merged_splitters_per_block);
    unsigned int dst_idx = block_idx << (log_num_merged_splitters_per_block + LOG_BLOCK_SIZE);

    if(threadIdx.x == 0)
    {
      resultdatakey[dst_idx] = srcdatakey[d_splitters_pos[splitter_idx]];
      resultdatavalue[dst_idx] = srcdatavalue[d_splitters_pos[splitter_idx]];
    } // end if
  } // end for
} // end copy_first_splitters()

///////////////////// Helper function to write out data in an aligned manner ////////////////////////////////////////
template<unsigned int BLOCK_SIZE, typename KeyType, typename ValueType>
  __device__ void aligned_write(KeyType * dest, ValueType * dest2,
                                KeyType * src, ValueType * src2,
                                unsigned int dest_offset, unsigned int elements)
{
  // copy from src to dest + dest_offset: dest, src are aligned, dest_offset not a multiple of 4
  int t = (int)threadIdx.x;
  unsigned int start_thread_aligned = dest_offset%WARP_SIZE;
  
  // write the first WARP_SIZE - start_thread_aligned elements
  if(t < WARP_SIZE && t >= start_thread_aligned && (t - start_thread_aligned < elements))
  {
    dest[dest_offset + t - start_thread_aligned] = src[t - start_thread_aligned];
    dest2[dest_offset + t - start_thread_aligned] = src2[t - start_thread_aligned];
  }
  
  // write upto BLOCK_SIZE elements in each iteration 
  unsigned int off = WARP_SIZE - start_thread_aligned;
  while(off + t < elements)
  {
    dest[dest_offset + off + t] = src[off + t]; 
    dest2[dest_offset + off + t] = src2[off + t]; 
    off+=BLOCK_SIZE;
  }
  __syncthreads();
}

///////////////////// Helper function to read in data in an aligned manner ////////////////////////////////////////
template<unsigned int BLOCK_SIZE, typename Tkey, typename Tvalue>
  __device__ void aligned_read(Tkey *dest, Tvalue *dest2,
                               const Tkey *src, const Tvalue *src2,
                               unsigned int src_offset,
                               unsigned int elements)
{
  // copy from src + src_offset to dest: dest, src are aligned, src_offset not a multiple of 4
  // copy from src2 + src_offset to dest2: dest2, src2 are aligned, src_offset not a multiple of 4
  
  int t = (int)threadIdx.x;
  unsigned int start_thread_aligned = src_offset%WARP_SIZE;
  
  // write the first WARP_SIZE - start_thread_aligned elements
  if(t < WARP_SIZE && t >= start_thread_aligned && (t - start_thread_aligned < elements))
  {
    dest[t - start_thread_aligned] = src[src_offset + t - start_thread_aligned];
    dest2[t - start_thread_aligned] = src2[src_offset + t - start_thread_aligned];
  }
  
  //write upto BLOCK_SIZE elements in each iteration 
  unsigned int off = WARP_SIZE - start_thread_aligned;
  while(off + t < elements)
  {
    dest[off + t] = src[src_offset + off + t]; 
    dest2[off + t] = src2[src_offset + off + t]; 
    off+=BLOCK_SIZE;
  }
  __syncthreads();
}

///////////////////// MERGE TWO INDEPENDENT SEGMENTS USING BINARY SEARCH IN SHARED MEMORY ////////////////////////////////////////
// NOTE: This is the most compute-intensive part of the algorithm. 
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Thread block i merges entries between rank[i] and rank[i+1]. These can be independently
// merged and concatenated, as noted above. 
// Each thread in the thread block i does a binary search of one element between rank[i] -> rank[i+1] in the 
// other array. 

// Inputs: srcdatakey, value: inputs
//         log_blocksize, log_num_merged_splitters_per_block: as in previous functions
// Outputs: resultdatakey, resultdatavalue: output merged arrays are written here.
template<unsigned int BLOCK_SIZE, unsigned int LOG_BLOCK_SIZE, typename KeyType, typename ValueType, typename StrictWeakOrdering>
__global__ void merge_subblocks_binarysearch_kernel(const KeyType * srcdatakey, const ValueType * srcdatavalue, unsigned int datasize, 
                                                    const unsigned int * rank1, const unsigned int * rank2, 
                                                    const unsigned int log_blocksize, 
                                                    const unsigned int log_num_merged_splitters_per_block, 
                                                    const unsigned int num_splitters,
                                                    KeyType *resultdatakey, ValueType *resultdatavalue,
                                                    StrictWeakOrdering comp)
{	
  extern __shared__ char A[];
  KeyType * input1 = (KeyType *)(A);
  KeyType * input2 = (KeyType *)(A + sizeof(KeyType)*BLOCK_SIZE);
  ValueType * input1val = (ValueType *)(A + sizeof(KeyType)*(2*BLOCK_SIZE));
  ValueType * input2val = (ValueType *)(A + sizeof(KeyType)*(2*BLOCK_SIZE) + sizeof(ValueType)*BLOCK_SIZE);
  
  // Thread Block i merges the sub-block associated with splitter i: rank[i] -> rank[i+1] in a particular odd-even block pair.
  for(unsigned int i = blockIdx.x;
      i < num_splitters;
      i += gridDim.x)
  {
    // the (odd, even) block pair that the splitter belongs to.
    unsigned int oddeven_blockid = i >> log_num_merged_splitters_per_block;
    
    // the index of the merged splitter within the splitters for the odd-even block pair.
    unsigned int local_blockIdx = i - (oddeven_blockid<<log_num_merged_splitters_per_block);
    
    // start1 & start2 store rank[i] and rank[i+1] indices in arrays 1 and 2.
    // size1 & size2 store the number of of elements between rank[i] & rank[i+1] in arrays 1 & 2.
    __shared__ unsigned int start1, start2, size1, size2;
    
    // thread 0 computes the ranks & size of each array
    if(threadIdx.x == 0)
    {
      start1 = rank1[i];
      start2 = rank2[i];

      // Carefully avoid out-of-bounds rank array accesses.
      if( (i < num_splitters - 1) && (local_blockIdx < ((1<<log_num_merged_splitters_per_block)-1)) )
      {
        size1 = rank1[i + 1];
        size2 = rank2[i + 1];
      } // end if
      else
      {
        size1 = size2 = (1<<log_blocksize);
      } // end else
      
      // Adjust size2 to account for the last block possibly not being full.
      if((size2 + (oddeven_blockid<<(log_num_merged_splitters_per_block + LOG_BLOCK_SIZE)) + (1<<log_blocksize)) 
         > datasize)
      {
        size2 = datasize - (1<<log_blocksize) - (oddeven_blockid<<(log_num_merged_splitters_per_block + LOG_BLOCK_SIZE));
      } // end if

      // measure each array relative to its beginning
      size1 -= start1;
      size2 -= start2;
    } // end if
    __syncthreads();
    
    // each block has to merge elements start1 - end1 of data1 with start2 - end2 of data2. 
    // We know that start1 - end1 < 2*CTASIZE, start2 - end2 < 2*CTASIZE
    const KeyType * local_srcdata1key = srcdatakey + (oddeven_blockid<<(log_num_merged_splitters_per_block + LOG_BLOCK_SIZE));
    const KeyType * local_srcdata2key = srcdatakey + (oddeven_blockid<<(log_num_merged_splitters_per_block + LOG_BLOCK_SIZE)) + (1<<log_blocksize);
    const ValueType * local_srcdata1value = srcdatavalue + (oddeven_blockid<<(log_num_merged_splitters_per_block + LOG_BLOCK_SIZE));
    const ValueType * local_srcdata2value = srcdatavalue + (oddeven_blockid<<(log_num_merged_splitters_per_block + LOG_BLOCK_SIZE)) + (1<<log_blocksize);
    
    // read in blocks
    // this causes unaligned loads to take place because start1 is usually unaligned.
    // We can do some fancy tricks to eliminate this unaligned load: somewhat better
    aligned_read<BLOCK_SIZE>(input1, input1val, local_srcdata1key, local_srcdata1value, start1, size1);
    
    // Read in other side
    aligned_read<BLOCK_SIZE>(input2, input2val, local_srcdata2key, local_srcdata2value, start2, size2);
    
    KeyType inp1 = input1[threadIdx.x]; ValueType inp1val = input1val[threadIdx.x];
    KeyType inp2 = input2[threadIdx.x]; ValueType inp2val = input2val[threadIdx.x];

    // this barrier is unnecessary for correctness but speeds up the kernel on G80
    __syncthreads();
    
    // to merge input1 and input2, use binary search to find the rank of inp1 & inp2 in arrays input2 & input1, respectively
    // as before, the "end" variables point to one element after the last element of the arrays
    unsigned int start_1, end_1, start_2, end_2, cur;

    // start by looking through input2 for inp1's rank
    start_1 = 0; end_1 = size2;
    
    // don't do the search if our value is beyond the end of input1
    if(threadIdx.x < size1)
    {
      while(start_1 < end_1)
      {
        cur = (start_1 + end_1)>>1;
        if(comp(input2[cur], inp1)) start_1 = cur + 1;
        else end_1 = cur;
      } // end while
    } // end if
    
    // now look through input1 for inp2's rank
    start_2 = 0; end_2 = size1;
    
    // don't do the search if our value is beyond the end of input2
    if(threadIdx.x < size2)
    {
      while(start_2 < end_2)
      {
        cur = (start_2 + end_2)>>1;
        // using two comparisons is a hack to break ties in such a way that input1 elements go before input2
        // XXX eliminate the need for two comparisons and make sure it is still stable
        KeyType temp = input1[cur];
        if(comp(temp, inp2) || !comp(inp2, temp)) start_2 = cur + 1;
        else end_2 = cur;
      } // end while
    } // end if
    __syncthreads();
    
    // Write back into the right position to the input arrays; can be done in place since we read in
    // the input arrays into registers before.
    if(threadIdx.x < size1)
    {
      input1[start_1 + threadIdx.x] = inp1;
      input1val[start_1 + threadIdx.x] = inp1val;
    } // end if
    
    if(threadIdx.x < size2)
    {
      input1[start_2 + threadIdx.x] = inp2;
      input1val[start_2 + threadIdx.x] = inp2val;
    } // end if
    __syncthreads();
    
    // Write out to global memory; we need to align the write for good performance
    aligned_write<BLOCK_SIZE>(resultdatakey, resultdatavalue, input1, input1val, (oddeven_blockid<<(log_num_merged_splitters_per_block + LOG_BLOCK_SIZE)) + start1 + start2, size1 + size2);
  } // end for i
} // end merge_subblocks_binarysearch_kernel()

// merge_subblocks_binarysearch() merges each sub-block independently. As explained in find_splitter_ranks(), 
// the sub-blocks are defined by the ranks of the splitter elements d_rank1 and d_rank2 in the odd and even blocks resp.
// It can be easily shown that each sub-block cannot contain more than BLOCK_SIZE elements of either the odd or even block.

// There are a total of (N_SPLITTERS + 1) sub-blocks for an odd-even block pair if the number of splitters is N_splitterS.
// Out of these sub-blocks, the very first sub-block for each odd-even block pair always contains only one element: 
// the smallest merged splitter. We just copy over this set of smallest merged splitters for each odd-even block pair
// to the output array before doing the merge of the remaining sub-blocks (copy_first_splitters()). 

// After this write, the function calls merge_subblocks_binarysearch_kernel() for the remaining N_splitterS sub-blocks
// We use 1 block per splitter: For instance, block 0 will merge rank1[0] -> rank1[1] of array i with
// rank2[0] -> rank2[1] of array i^1, with i being the block to which the splitter belongs.

// We implement each sub-block merge using a binary search. We compute the rank of each element belonging to a sub-block 
// of an odd numbered block in the corresponding sub-block of its even numbered pair. It then adds this rank to 
// the index of the element in its own sub-block to find the output index of the element in the merged sub-block.

template<typename KeyType, typename ValueType, typename StrictWeakOrdering>
  void merge_subblocks_binarysearch(KeyType * srcdatakey, ValueType * srcdatavalue, unsigned int datasize, 
                                    KeyType * d_splitters, unsigned int * d_splitters_pos, 
                                    unsigned int * rank1, unsigned int * rank2, 
                                    KeyType *resultdatakey, ValueType *resultdatavalue, 
                                    unsigned int N_SPLITTERS, unsigned int log_blocksize, 
                                    unsigned int log_num_merged_splitters_per_block,
                                    unsigned int N_ODDEVEN_BLOCK_PAIRS,
                                    StrictWeakOrdering comp)
{
  unsigned int MAX_GRID_SIZE = max_grid_size(1);

  unsigned int grid_size = min(N_ODDEVEN_BLOCK_PAIRS, MAX_GRID_SIZE);

  const unsigned int LOG_BLOCK_SIZE = merge_sort_dev_namespace::LOG_BLOCK_SIZE<KeyType,ValueType>::result;

  // Copy over the first merged splitter of each odd-even block pair to the output array
  copy_first_splitters<LOG_BLOCK_SIZE><<<N_ODDEVEN_BLOCK_PAIRS,1>>>(srcdatakey, srcdatavalue, d_splitters_pos, 
                                                                    resultdatakey, resultdatavalue,
                                                                    log_num_merged_splitters_per_block,
                                                                    N_ODDEVEN_BLOCK_PAIRS);

  const unsigned int BLOCK_SIZE = merge_sort_dev_namespace::BLOCK_SIZE<KeyType,ValueType>::result;

  MAX_GRID_SIZE = max_grid_size(BLOCK_SIZE);

  grid_size = min(N_SPLITTERS, MAX_GRID_SIZE);

  merge_subblocks_binarysearch_kernel<BLOCK_SIZE,LOG_BLOCK_SIZE><<<grid_size, BLOCK_SIZE, BLOCK_SIZE*(2*sizeof(KeyType) + 2*sizeof(ValueType))>>>(
  	srcdatakey, srcdatavalue, datasize, 
  	rank1, rank2, 
  	log_blocksize, log_num_merged_splitters_per_block, 
        N_SPLITTERS,
  	resultdatakey, resultdatavalue,
        comp);
}

template<typename KeyType,
         typename ValueType,
         typename StrictWeakOrdering>
  void merge(KeyType *keys_src,
             ValueType *data_src,
             size_t n,
             KeyType *keys_dst,
             ValueType *data_dst,
             size_t log_tile_size,
             size_t level,
             StrictWeakOrdering comp)
{
  size_t tile_size = 1<<log_tile_size;

  // assumption: num_tiles is even; tile_size is a power of 2
  size_t num_tiles = n / tile_size;
  size_t partial_tile_size = n % tile_size;
  if(partial_tile_size) ++num_tiles;

  // Compute the BLOCK_SIZE based on the types to sort
  const unsigned int BLOCK_SIZE = merge_sort_dev_namespace::BLOCK_SIZE<KeyType,ValueType>::result;

  // Case (a): tile_size <= BLOCK_SIZE
  if(tile_size <= BLOCK_SIZE)
  {
    // Two or more tiles can fully fit into shared memory, and can be merged by one thread block.
    // In particular, we load (2*BLOCK_SIZE) elements into shared memory, 
    //        and merge all the contained tile pairs using one thread block.   
    // We use (2*BLOCK_SIZE) threads/thread block and grid_size * tile_size/(2*BLOCK_SIZE) thread blocks.
    unsigned int tiles_per_block = (2*BLOCK_SIZE) / tile_size;
    unsigned int partial_block_size = num_tiles % tiles_per_block;
    unsigned int number_of_tiles_in_last_block = partial_block_size ? partial_block_size : tiles_per_block;
    unsigned int num_blocks = num_tiles / tiles_per_block;
    if(partial_block_size) ++num_blocks;

    // compute the maximum number of blocks we can launch on this arch
    const unsigned int MAX_GRID_SIZE = max_grid_size(2 * BLOCK_SIZE);
    unsigned int grid_size = min(num_blocks, MAX_GRID_SIZE);

    // figure out the size & index of the last tile of the last block
    unsigned int size_of_last_tile = partial_tile_size ? partial_tile_size : tile_size;
    unsigned int index_of_last_tile_in_last_block = number_of_tiles_in_last_block - 1;

    merge_smalltiles_binarysearch<2*BLOCK_SIZE><<<grid_size,(2*BLOCK_SIZE)>>>(keys_src, data_src,
                                                                            n,
                                                                            num_blocks - 1,
                                                                            index_of_last_tile_in_last_block,
                                                                            size_of_last_tile,
                                                                            keys_dst, data_dst,
                                                                            log_tile_size,
                                                                            comp);

    return;
  } // end if

  // Case (b) tile_size >= BLOCK_SIZE

  // compute the maximum number of blocks we can launch on this arch
  const unsigned int MAX_GRID_SIZE = max_grid_size(BLOCK_SIZE);

  // Step 1 of the recursive case: extract one splitter per BLOCK_SIZE entries in each odd-even block pair.
  // Store the splitter keys into splitters[level], and the array index in keys_src of the splitters
  // chosen into splitters_pos[level]
  size_t num_splitters = n / BLOCK_SIZE;
  if(n % BLOCK_SIZE) ++num_splitters;

  unsigned int grid_size = min<size_t>(num_splitters, MAX_GRID_SIZE);

  // XXX replace these with scoped_ptr or something
  thrust::device_ptr<KeyType>      splitters            = device_malloc<KeyType>(num_splitters);
  thrust::device_ptr<unsigned int> splitters_pos        = device_malloc<unsigned int>(num_splitters);
  thrust::device_ptr<KeyType>      merged_splitters     = device_malloc<KeyType>(num_splitters);
  thrust::device_ptr<unsigned int> merged_splitters_pos = device_malloc<unsigned int>(num_splitters);

  extract_splitters<KeyType><<<grid_size, BLOCK_SIZE>>>(keys_src, n, splitters.get(), splitters_pos.get());

  // compute the log base 2 of the BLOCK_SIZE
  const unsigned int LOG_BLOCK_SIZE = merge_sort_dev_namespace::LOG_BLOCK_SIZE<KeyType,ValueType>::result;

  // Step 2 of the recursive case: merge these elements using merge
  // We need to merge num_splitters elements, each new "block" is the set of
  // splitters for each original block.
  size_t log_num_splitters_per_block = log_tile_size - LOG_BLOCK_SIZE;
  merge<KeyType, unsigned int, StrictWeakOrdering>
    (splitters.get(), splitters_pos.get(),
     num_splitters,
     merged_splitters.get(), merged_splitters_pos.get(),
     log_num_splitters_per_block,
     level + 1, comp);
  device_free(splitters);

  // Step 3 of the recursive case: Find the ranks of each splitter in the respective two blocks.
  // Store the results into rank1[level] and rank2[level] for the even and odd block respectively.
  // rank1[level] and rank2[level] define the sub-block splits:
  //      Sub-block 0: Elements with indices less than rank1[0] in the odd block less than rank2[0] in the even
  //      Sub-block 1: Indices between rank1[0] and rank1[1] in the odd block and
  //                           between rank2[0] and rank2[1] in the even block
  //      ... and so on.
  size_t log_num_merged_splitters_per_block = log_num_splitters_per_block + 1;

  size_t num_blocks = num_splitters / BLOCK_SIZE;
  if(num_splitters % BLOCK_SIZE) ++num_blocks;

  grid_size = min<size_t>(num_blocks, MAX_GRID_SIZE);

  // reuse the splitters_pos storage for rank1
  thrust::device_ptr<unsigned int> rank1 = splitters_pos;
  thrust::device_ptr<unsigned int> rank2 = device_malloc<unsigned int>(num_splitters);

  find_splitter_ranks<BLOCK_SIZE, LOG_BLOCK_SIZE, KeyType, KeyType, StrictWeakOrdering>
    <<<grid_size,BLOCK_SIZE>>>
      (merged_splitters.get(), merged_splitters_pos.get(),
           rank1.get(),         rank2.get(),
       keys_src,    n,
       num_splitters, log_tile_size,
       log_num_merged_splitters_per_block, comp);

  // Step 4 of the recursive case: merge each sub-block independently in parallel.
  merge_subblocks_binarysearch<KeyType, ValueType>(keys_src, data_src, n,
                                                   merged_splitters.get(), merged_splitters_pos.get(),
                                                       rank1.get(),         rank2.get(),
                                                   keys_dst, data_dst,
                                                   num_splitters, log_tile_size,
                                                   log_num_merged_splitters_per_block,
                                                   num_tiles / 2,
                                                   comp);

  device_free(merged_splitters);
  device_free(merged_splitters_pos);
  device_free(rank1);
  device_free(rank2);
}

template<typename KeyType, typename ValueType, typename StrictWeakOrdering>
  void merge(KeyType *keys_src,
             ValueType *data_src,
             size_t n,
             KeyType *keys_dst,
             ValueType *data_dst,
             size_t block_size,
             StrictWeakOrdering comp)
{
  unsigned int log_block_size = (unsigned int)logb((float)block_size);
  unsigned int NBLOCKS = (n%block_size)?((n/block_size)+1):(n/block_size);

  merge(keys_src, data_src, (NBLOCKS%2)?((NBLOCKS-1)*block_size):n, keys_dst, data_dst, log_block_size, 0, comp);

  if(NBLOCKS%2)
  {
    thrust::copy(device_pointer_cast(keys_src) + (NBLOCKS-1)*block_size,
                 device_pointer_cast(keys_src) + n,
                 device_pointer_cast(keys_dst) + (NBLOCKS-1)*block_size);
    
    thrust::copy(device_pointer_cast(data_src) + (NBLOCKS-1)*block_size,
                 device_pointer_cast(data_src) + n,
                 device_pointer_cast(data_dst) + (NBLOCKS-1)*block_size);
  }
}


} // end merge_sort_dev_namespace

template<typename KeyType,
         typename ValueType,
         typename StrictWeakOrdering>
  void stable_merge_sort_by_key_dev(KeyType *keys,
                                    ValueType *data,
                                    StrictWeakOrdering comp,
                                    const size_t n)
{
  // don't launch an empty kernel
  if(n == 0) return;

  // compute the BLOCK_SIZE based on the types we're sorting
  const unsigned int BLOCK_SIZE = merge_sort_dev_namespace::BLOCK_SIZE<KeyType,ValueType>::result;

  // compute the maximum number of blocks we can launch on this arch
  const unsigned int MAX_GRID_SIZE = merge_sort_dev_namespace::max_grid_size(BLOCK_SIZE);

  // first, sort within each block
  size_t num_blocks = n / BLOCK_SIZE;
  if(n % BLOCK_SIZE) ++num_blocks;

  size_t grid_size = merge_sort_dev_namespace::min<size_t>(num_blocks, MAX_GRID_SIZE);

  // do an odd-even sort per block of data
  merge_sort_dev_namespace::stable_odd_even_block_sort_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(keys, data, comp, n);

  // scratch space
  thrust::device_ptr<KeyType>   temp_keys = device_malloc<KeyType>(n);
  thrust::device_ptr<ValueType> temp_data = device_malloc<ValueType>(n);

  KeyType   *keys0 = keys, *keys1 = temp_keys.get();
  ValueType *data0 = data, *data1 = temp_data.get();

  // The log(n) iterations start here. Each call to 'merge' merges an odd-even pair of tiles
  // Currently uses additional arrays for sorted outputs.
  unsigned int iterations = 0;
  for(unsigned int tile_size = BLOCK_SIZE;
      tile_size < n;
      tile_size *= 2)
  {
    merge_sort_dev_namespace::merge(keys0, data0, n, keys1, data1, tile_size, comp);
    thrust::swap(keys0, keys1);
    thrust::swap(data0, data1);
    ++iterations;
  }

  // this is to make sure that our data is finally in the data and keys arrays
  // and not in the temporary arrays
  if(iterations % 2)
  {
    thrust::copy(device_pointer_cast(data0),
                 device_pointer_cast(data0 + n),
                 device_pointer_cast(data1));
    thrust::copy(device_pointer_cast(keys0),
                 device_pointer_cast(keys0 + n),
                 device_pointer_cast(keys1));
  }

  device_free(temp_keys);
  device_free(temp_data);
} // end stable_merge_sort_by_key_dev()


} // end namespace cuda

} // end namespace device

} // end namespace detail

} // end namespace sorting

} // end namespace thrust

#endif // __CUDACC__

