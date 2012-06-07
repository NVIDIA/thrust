/*
 *  Copyright 2008-2012 NVIDIA Corporation
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

#include <thrust/detail/config.h>

// do not attempt to compile this file with any other compiler
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#include <thrust/functional.h>
#include <thrust/detail/copy.h>
#include <thrust/swap.h>

#include <thrust/device_ptr.h>
#include <thrust/detail/backend/dereference.h>

#include <thrust/detail/mpl/math.h> // for log2<N>
#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/backend/cuda/block/merging_sort.h>
#include <thrust/detail/backend/cuda/synchronize.h>
#include <thrust/detail/backend/cuda/arch.h>
#include <thrust/detail/uninitialized_array.h>



__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN


namespace thrust
{
namespace detail
{

template<typename T>
  void destroy(T &x)
{
  x.~T();
} // end destroy()

namespace backend
{
namespace cuda
{
namespace detail
{

namespace merge_sort_dev_namespace
{

// define our own min() rather than #include <thrust/extrema.h>
template<typename T>
  __host__ __device__
  T min THRUST_PREVENT_MACRO_SUBSTITUTION (const T &lhs, const T &rhs)
{
  return lhs < rhs ? lhs : rhs;
} // end min()


template<typename Key, typename Value>
  class block_size
{
  private:
    static const unsigned int sizeof_larger_type = (sizeof(Key) > sizeof(Value) ? sizeof(Key) : sizeof(Value));

    static const unsigned int max_smem_usage = 2048;
    static const unsigned int max_blocksize = 128;

    // our block size candidate is simply 2K over the sum of sizes
    static const unsigned int candidate = max_smem_usage / (sizeof(Key) + sizeof(Value));

    // round one_k_over_size down to the nearest power of two
    static const unsigned int lg_candidate = thrust::detail::mpl::math::log2<candidate>::value;

    // exponentiate that result, which rounds down to the nearest power of two
    static const unsigned int final_candidate = 1<<lg_candidate;

  public:
    // XXX WAR nvcc 3.0 bug
    //static const unsigned int result =  (final_candidate < max_blocksize) ? final_candidate : max_blocksize;

    static const unsigned int result = 
      thrust::detail::eval_if<
        final_candidate < max_blocksize,
        thrust::detail::integral_constant<unsigned int, final_candidate>,
        thrust::detail::integral_constant<unsigned int, max_blocksize>
      >::type::value;
};

template<typename Key, typename Value>
  class log_block_size
{
  private:
    static const unsigned int bs = block_size<Key,Value>::result;

  public:
    static const unsigned int result = thrust::detail::mpl::math::log2<bs>::value;
};

static const unsigned int warp_size = 32;

inline unsigned int max_grid_size(const unsigned int block_size)
{
  const arch::device_properties_t& properties = thrust::detail::backend::cuda::arch::device_properties();

  const unsigned int max_threads = properties.maxThreadsPerMultiProcessor * properties.multiProcessorCount;
  const unsigned int max_blocks  = properties.maxGridSize[0];
  
  return std::min<unsigned int>(max_blocks, 3 * max_threads / block_size);
} // end max_grid_size()

template<unsigned int N>
  struct align_size_to_int
{
  static const unsigned int value = (N / sizeof(int)) + ((N % sizeof(int)) ? 1 : 0);
};

// Base case for the merge algorithm: merges data where tile_size <= block_size. 
// Works by loading two or more tiles into shared memory and doing a binary search.
template<unsigned int block_size,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename StrictWeakOrdering>
__global__ void merge_smalltiles_binarysearch(RandomAccessIterator1 keys_first,
                                              RandomAccessIterator2 values_first,
                                              const unsigned int n,
                                              const unsigned int index_of_last_block,
                                              const unsigned int index_of_last_tile_in_last_block,
                                              const unsigned int size_of_last_tile,
                                              RandomAccessIterator3 keys_result,
                                              RandomAccessIterator4 values_result,
                                              const unsigned int log_tile_size,
                                              StrictWeakOrdering comp)
{
  using namespace thrust::detail::backend;

  typedef typename iterator_value<RandomAccessIterator3>::type KeyType;
  typedef typename iterator_value<RandomAccessIterator4>::type ValueType;

  // Assumption: tile_size is a power of 2.

  // load (2*block_size) elements into shared memory. These (2*block_size) elements belong to (2*block_size)/tile_size different tiles.
  // use int for these shared arrays due to alignment issues
  __shared__ int key_workaround[align_size_to_int<2 * block_size * sizeof(KeyType)>::value];
  KeyType *key = reinterpret_cast<KeyType*>(key_workaround);

  __shared__ int outkey_workaround[align_size_to_int<2 * block_size * sizeof(KeyType)>::value];
  KeyType *outkey = reinterpret_cast<KeyType*>(outkey_workaround);

  __shared__ int outvalue_workaround[align_size_to_int<2 * block_size * sizeof(ValueType)>::value];
  ValueType *outvalue = reinterpret_cast<ValueType*>(outvalue_workaround);

  const unsigned int grid_size = gridDim.x * blockDim.x;

  // this will store the final rank of our key
  unsigned int rank;

  unsigned int block_idx = blockIdx.x;
  
  // the global index of this thread
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

  // advance iterators
  keys_first    += i;
  values_first  += i;
  keys_result   += i;
  values_result += i;

  for(;
      block_idx <= index_of_last_block;
      block_idx += gridDim.x, i += grid_size, keys_first += grid_size, values_first += grid_size, keys_result += grid_size, values_result += grid_size)
  {
    // figure out if this thread should idle
    unsigned int thread_not_idle = i < n;
    KeyType my_key;
    
    // copy over inputs to shared memory
    if(thread_not_idle)
    {
      key[threadIdx.x] = my_key = dereference(keys_first);
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
    KeyType * other = key + (other_tile_index<<log_tile_size);
    
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
      outvalue[rank] = dereference(values_first);
    } // end if
    __syncthreads();
    
    if(thread_not_idle)
    {
      // coalesced writes to global memory
      dereference(keys_result)   = outkey[threadIdx.x];
      dereference(values_result) = outvalue[threadIdx.x];
    } // end if
    __syncthreads();
  } // end for
} // end merge_smalltiles_binarysearch()

template<unsigned int block_size,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  __global__ void stable_odd_even_block_sort_kernel(RandomAccessIterator1 keys_first,
                                                    RandomAccessIterator2 values_first,
                                                    StrictWeakOrdering comp,
                                                    const unsigned int n)
{
  using namespace thrust::detail::backend;

  typedef typename iterator_value<RandomAccessIterator1>::type KeyType;
  typedef typename iterator_value<RandomAccessIterator2>::type ValueType;

  // XXX workaround no constructors on device arrays
//  __shared__ unsigned char s_keys_workaround[block_size * sizeof(KeyType)];
//  KeyType *s_keys = reinterpret_cast<KeyType*>(s_keys_workaround);
//
//  __shared__ unsigned char s_data_workaround[block_size * sizeof(ValueType)];
//  ValueType *s_data = reinterpret_cast<ValueType*>(s_data_workaround);
  __shared__ int s_keys_workaround[align_size_to_int<block_size * sizeof(KeyType)>::value];
  KeyType *s_keys = reinterpret_cast<KeyType*>(s_keys_workaround);

  __shared__ int s_data_workaround[align_size_to_int<block_size * sizeof(ValueType)>::value];
  ValueType *s_data = reinterpret_cast<ValueType*>(s_data_workaround);

  const unsigned int grid_size = gridDim.x * blockDim.x;

  // block_offset records the global index of this block's 0th thread
  unsigned int block_offset = blockIdx.x * block_size;
  unsigned int i = threadIdx.x + block_offset;

  // advance iterators
  keys_first   += i;
  values_first += i;

  for(;
      block_offset < n;
      block_offset += grid_size, i += grid_size, keys_first += grid_size, values_first += grid_size)
  {
    __syncthreads();
    // copy input to shared
    if(i < n)
    {
      s_keys[threadIdx.x] = dereference(keys_first);
      s_data[threadIdx.x] = dereference(values_first);
    } // end if
    __syncthreads();

    // this block could be partially full
    unsigned int length = block_size;
    if(block_offset + block_size > n)
    {
      length = n - block_offset;
    } // end if

    // run merge_sort over the block
    thrust::detail::backend::cuda::block::merging_sort(s_keys, s_data, length, comp);

    // write result
    if(i < n)
    {
      dereference(keys_first)   = s_keys[threadIdx.x];
      dereference(values_first) = s_data[threadIdx.x];
    } // end if
  } // end for i
} // end stable_odd_even_block_sort_kernel()

// Extract the splitters: every blockDim.x'th element of src is a splitter
// Input: src, src_size
// Output: splitters: the splitters
//         splitters_pos: Index of splitter in src
template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3>
  __global__ void extract_splitters(RandomAccessIterator1 first,
                                    unsigned int N,
                                    RandomAccessIterator2 splitters_result,
                                    RandomAccessIterator3 positions_result)
{
  using namespace thrust::detail::backend;

  const unsigned int grid_size = gridDim.x * blockDim.x;

  unsigned int i = blockIdx.x * blockDim.x;

  // advance iterators
  splitters_result += blockIdx.x;
  positions_result += blockIdx.x;
  first            += blockIdx.x * blockDim.x;

  for(;
      i < N;
      i += grid_size, splitters_result += gridDim.x, positions_result += gridDim.x, first += grid_size)
  {
    if(threadIdx.x == 0)
    {
      dereference(splitters_result) = dereference(first); 
      dereference(positions_result) = i;
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
template<unsigned int block_size,
         unsigned int log_block_size,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename RandomAccessIterator5,
         typename StrictWeakOrdering>
  __global__ void find_splitter_ranks(RandomAccessIterator1 splitters_first,
                                      RandomAccessIterator2 splitters_pos_first, 
                                      RandomAccessIterator3 ranks_result1,
                                      RandomAccessIterator4 ranks_result2, 
                                      RandomAccessIterator5 values_begin,
                                      unsigned int datasize, 
                                      unsigned int num_splitters,
                                      unsigned int log_tile_size, 
                                      unsigned int log_num_merged_splitters_per_block,
                                      StrictWeakOrdering comp)
{
  using namespace thrust::detail::backend;

  typedef typename iterator_value<RandomAccessIterator1>::type KeyType;
  typedef typename iterator_value<RandomAccessIterator2>::type IndexType;

  KeyType inp;
  IndexType inp_pos;
  int start, end, cur;

  const unsigned int grid_size = gridDim.x * blockDim.x;

  unsigned int block_offset = blockIdx.x * blockDim.x;
  unsigned int i = threadIdx.x + block_offset;

  // advance iterators
  splitters_first     += i;
  splitters_pos_first += i;
  ranks_result1       += i;
  ranks_result2       += i;
  
  for(;
      block_offset < num_splitters;
      block_offset += grid_size, i += grid_size, splitters_first += grid_size, splitters_pos_first += grid_size, ranks_result1 += grid_size, ranks_result2 += grid_size)
  {
    if(i < num_splitters)
    { 
      inp     = dereference(splitters_first);
      inp_pos = dereference(splitters_pos_first);
      
      // the (odd, even) block pair to which the splitter belongs. Each i corresponds to a splitter.
      unsigned int oddeven_blockid = i>>log_num_merged_splitters_per_block;
      
      // the local index of the splitter within the (odd, even) tile pair.
      unsigned int local_i  = i- ((oddeven_blockid)<<log_num_merged_splitters_per_block);
      
      // the tile to which the splitter belongs.
      unsigned int listno = (inp_pos >> log_tile_size);
      
      // the "other" block which which block listno must be merged.
      unsigned int otherlist = listno^1;
      RandomAccessIterator5 other = values_begin + (1<<log_tile_size)*otherlist;
      
      // the size of the other block can be less than blocksize if the it is the last block.
      unsigned int othersize = min<unsigned int>(1 << log_tile_size, datasize - (otherlist<<log_tile_size));
      
      // We want to compute the ranks of element inp in d_srcData1 and d_srcData2
      // for instance, if inp belongs to d_srcData1, then 
      // (1) the rank in d_srcData1 is simply given by its inp_pos
      // (2) to find the rank in d_srcData2, we first find the block in d_srcData2 where inp appears.
      //     We do this by noting that we have already merged/sorted splitters, and thus the rank
      //     of inp in the elements of d_srcData2 that are present in splitters is given by 
      //        position of inp in d_splitters - rank of inp in elements of d_srcData1 in splitters
      //        = i - inp_pos
      //     This also gives us the block of d_srcData2 that inp belongs in, since we have one
      //     element in splitters per block of d_srcData2.
      
      //     We now perform a binary search over this block of d_srcData2 to find the rank of inp in d_srcData2.
      //     start and end are the start and end indices of this block in d_srcData2, forming the bounds of the binary search.
      //     Note that this binary search is in global memory with uncoalesced loads. However, we only find the ranks 
      //     of a small set of elements, one per splitter: thus it is not the performance bottleneck.
      if(!(listno&0x1))
      { 
        dereference(ranks_result1) = inp_pos + 1 - (1<<log_tile_size)*listno; 

        // XXX this is a redundant load
        end = (( local_i - ((dereference(ranks_result1) - 1)>>log_block_size)) << log_block_size ) - 1;
        start = end - (block_size-1);

        if(end < 0) start = end = 0;
        if(end >= othersize) end = othersize - 1;
        if(start > othersize) start = othersize;
      } // end if
      else
      { 
        dereference(ranks_result2) = inp_pos + 1 - (1<<log_tile_size)*listno;

        // XXX this is a redundant load
        end = (( local_i - ((dereference(ranks_result2) - 1)>>log_block_size)) << log_block_size ) - 1;
        start = end - (block_size-1);

        if(end < 0) start = end = 0;
        if(end >= othersize) end = othersize - 1;
        if(start > othersize) start = othersize;
      } // end else
      
      // XXX we need to implement this section with lower_bound()
      // we have the start and end indices for the binary search in the "other" array
      // do a binary search. Break ties by letting elements of array1 before those of array2 
      while(start <= end)
      {
        cur = (start + end)>>1;
        RandomAccessIterator5 mid = other + cur;

        // XXX eliminate the need for two comparisons here and ensure the sort is still stable
        // XXX this is a redundant load
        if((comp(dereference(mid), inp))
           || (!comp(inp, dereference(mid)) && (listno&0x1)))
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
        dereference(ranks_result2) = start;	
      } // end if
      else
      {
        dereference(ranks_result1) = start;	
      } // end else
    } // end if
  } // end for
} // end find_splitter_ranks()

///////////////// Copy over first merged splitter of each odd-even block pair to the output array //////////////////
template<unsigned int log_tile_size,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename RandomAccessIterator5>
  __global__ void copy_first_splitters(RandomAccessIterator1 keys_first,
                                       RandomAccessIterator2 values_first,
                                       RandomAccessIterator3 splitters_pos_first, 
                                       RandomAccessIterator4 keys_result, 
                                       RandomAccessIterator5 values_result,
                                       unsigned int log_num_merged_splitters_per_block,
                                       const unsigned int num_tile_pairs)
{
  using namespace thrust::detail::backend;

  unsigned int num_splitters_per_tile  = 1 << (log_num_merged_splitters_per_block);
  unsigned int num_splitters_per_block = 1 << (log_num_merged_splitters_per_block + log_tile_size);
  unsigned int num_splitters_per_grid  = num_splitters_per_block * gridDim.x;

  unsigned int block_idx = blockIdx.x;

  // advance iterators
  keys_result         += block_idx * num_splitters_per_block;
  values_result       += block_idx * num_splitters_per_block;
  splitters_pos_first += block_idx * num_splitters_per_tile;

  for(;
      block_idx < num_tile_pairs;
      block_idx += gridDim.x, keys_result += num_splitters_per_grid, values_result += num_splitters_per_grid,
      splitters_pos_first += num_splitters_per_tile * gridDim.x)
  {
    if(threadIdx.x == 0)
    {
      // read in the splitter position once
      typename thrust::iterator_value<RandomAccessIterator3>::type splitter_pos = dereference(splitters_pos_first);

      RandomAccessIterator1 key_iter   = keys_first + splitter_pos;
      RandomAccessIterator2 value_iter = values_first + splitter_pos;

      dereference(keys_result)   = dereference(key_iter);
      dereference(values_result) = dereference(value_iter);
    } // end if
  } // end for
} // end copy_first_splitters()

// XXX we should eliminate this in favor of block::copy()
///////////////////// Helper function to write out data in an aligned manner ////////////////////////////////////////
template<unsigned int block_size,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4>
  __device__ void aligned_write(RandomAccessIterator1 first1,
                                RandomAccessIterator2 first2,
                                RandomAccessIterator3 result1,
                                RandomAccessIterator4 result2,
                                unsigned int dest_offset,
                                unsigned int num_elements)
{
  using namespace thrust::detail::backend;

  // copy from src to dest + dest_offset: dest, src are aligned, dest_offset not a multiple of 4
  unsigned int start_thread_aligned = dest_offset%warp_size;
  
  // write the first warp_size - start_thread_aligned elements
  if(threadIdx.x < warp_size && threadIdx.x >= start_thread_aligned && (threadIdx.x - start_thread_aligned < num_elements))
  {
    unsigned int offset = threadIdx.x - start_thread_aligned;

    // carefully create these iterator without creating temporary objects
    RandomAccessIterator1 first1_temp  = first1;
    first1_temp += offset;

    RandomAccessIterator2 first2_temp  = first2;
    first2_temp += offset;

    RandomAccessIterator3 result1_temp = result1;
    result1_temp += offset + dest_offset;

    RandomAccessIterator4 result2_temp = result2;
    result2_temp += offset + dest_offset;


    dereference(result1_temp) = dereference(first1_temp);
    dereference(result2_temp) = dereference(first2_temp);
  }

  int i = warp_size - start_thread_aligned + threadIdx.x; 

  // advance iterators
  first1  += i;
  first2  += i;
  result1 += i + dest_offset;
  result2 += i + dest_offset;
  
  // write upto block_size elements in each iteration 
  for(;
      i < num_elements;
      i += block_size, first1 += block_size, first2 += block_size, result1 += block_size, result2 += block_size)
  {
    dereference(result1) = dereference(first1); 
    dereference(result2) = dereference(first2); 
  }
  __syncthreads();
}

// XXX we should eliminate this in favor of block::copy
///////////////////// Helper function to read in data in an aligned manner ////////////////////////////////////////
template<unsigned int block_size,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4>
  __device__ void aligned_read(RandomAccessIterator1 first1,
                               RandomAccessIterator2 first2,
                               RandomAccessIterator3 result1,
                               RandomAccessIterator4 result2,
                               unsigned int src_offset,
                               unsigned int num_elements)
{
  using namespace thrust::detail::backend;

  // copy from src + src_offset to dest: dest, src are aligned, src_offset not a multiple of 4
  unsigned int start_thread_aligned = src_offset%warp_size;
  
  // write the first warp_size - start_thread_aligned elements
  if((threadIdx.x < warp_size) &&
     (threadIdx.x >= start_thread_aligned) &&
     ((threadIdx.x - start_thread_aligned) < num_elements))
  {
      // carefully create these iterators without causing creation of temporary objects
      unsigned int offset = threadIdx.x - start_thread_aligned;

      RandomAccessIterator1 first1_temp  = first1;
      first1_temp += offset + src_offset;

      RandomAccessIterator2 first2_temp  = first2;
      first2_temp += offset + src_offset;

      RandomAccessIterator3 result1_temp = result1;
      result1_temp += offset;

      RandomAccessIterator4 result2_temp = result2;
      result2_temp += offset;

      dereference(result1_temp) = dereference(first1_temp);
      dereference(result2_temp) = dereference(first2_temp);
  }

  int i = warp_size - start_thread_aligned + threadIdx.x;

  // advance iterators
  first1  += i + src_offset;
  first2  += i + src_offset;
  result1 += i;
  result2 += i;
  
  // write up to block_size elements in each iteration 
  for(;
      i < num_elements;
      i += block_size, first1 + block_size, first2 += block_size, result1 += block_size, result2 += block_size)
  {
    dereference(result1) = dereference(first1);
    dereference(result2) = dereference(first2);
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
template<unsigned int block_size,
         unsigned int log_block_size,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename RandomAccessIterator5,
         typename RandomAccessIterator6,
         typename StrictWeakOrdering>
__global__ void merge_subblocks_binarysearch_kernel(RandomAccessIterator1 keys_first,
                                                    RandomAccessIterator2 values_first,
                                                    unsigned int datasize, 
                                                    RandomAccessIterator3 ranks_first1,
                                                    RandomAccessIterator4 ranks_first2, 
                                                    const unsigned int log_tile_size, 
                                                    const unsigned int log_num_merged_splitters_per_block, 
                                                    const unsigned int num_splitters,
                                                    RandomAccessIterator5 keys_result,
                                                    RandomAccessIterator6 values_result,
                                                    StrictWeakOrdering comp)
{	
  using namespace thrust::detail::backend;

  typedef typename iterator_value<RandomAccessIterator5>::type KeyType;
  typedef typename iterator_value<RandomAccessIterator6>::type ValueType;

//  extern __shared__ char A[];
//  KeyType * input1 = (KeyType *)(A);
//  KeyType * input2 = (KeyType *)(A + sizeof(KeyType)*block_size);
//  ValueType * input1val = (ValueType *)(A + sizeof(KeyType)*(2*block_size));
//  ValueType * input2val = (ValueType *)(A + sizeof(KeyType)*(2*block_size) + sizeof(ValueType)*block_size);

  // use int for these shared arrays due to alignment issues
  __shared__ int input1_workaround[align_size_to_int<block_size * sizeof(KeyType)>::value];
  KeyType *input1 = reinterpret_cast<KeyType*>(input1_workaround);

  __shared__ int input2_workaround[align_size_to_int<block_size * sizeof(KeyType)>::value];
  KeyType *input2 = reinterpret_cast<KeyType*>(input2_workaround);

  __shared__ int input1val_workaround[align_size_to_int<block_size * sizeof(ValueType)>::value];
  ValueType *input1val = reinterpret_cast<ValueType*>(input1val_workaround);

  __shared__ int input2val_workaround[align_size_to_int<block_size * sizeof(ValueType)>::value];
  ValueType *input2val = reinterpret_cast<ValueType*>(input2val_workaround);


  // advance iterators
  unsigned int i = blockIdx.x;
  ranks_first1 += i;
  ranks_first2 += i;
  
  // Thread Block i merges the sub-block associated with splitter i: rank[i] -> rank[i+1] in a particular odd-even block pair.
  for(;
      i < num_splitters;
      i += gridDim.x, ranks_first1 += gridDim.x, ranks_first2 += gridDim.x)
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
      start1 = dereference(ranks_first1);
      start2 = dereference(ranks_first2);

      // Carefully avoid out-of-bounds rank array accesses.
      if( (i < num_splitters - 1) && (local_blockIdx < ((1<<log_num_merged_splitters_per_block)-1)) )
      {
        RandomAccessIterator3 temp1 = ranks_first1 + 1;
        RandomAccessIterator4 temp2 = ranks_first2 + 1;

        size1 = dereference(temp1);
        size2 = dereference(temp2);
      } // end if
      else
      {
        size1 = size2 = (1<<log_tile_size);
      } // end else
      
      // Adjust size2 to account for the last block possibly not being full.
      if((size2 + (oddeven_blockid<<(log_num_merged_splitters_per_block + log_block_size)) + (1<<log_tile_size)) 
         > datasize)
      {
        size2 = datasize - (1<<log_tile_size) - (oddeven_blockid<<(log_num_merged_splitters_per_block + log_block_size));
      } // end if

      // measure each array relative to its beginning
      size1 -= start1;
      size2 -= start2;
    } // end if
    __syncthreads();

    
    // each block has to merge elements start1 - end1 of data1 with start2 - end2 of data2. 
    // We know that start1 - end1 < 2*CTASIZE, start2 - end2 < 2*CTASIZE
    RandomAccessIterator1 local_keys_first1   = keys_first   + (oddeven_blockid<<(log_num_merged_splitters_per_block + log_block_size));
    RandomAccessIterator1 local_keys_first2   = local_keys_first1   + (1<<log_tile_size);

    RandomAccessIterator2 local_values_first1 = values_first + (oddeven_blockid<<(log_num_merged_splitters_per_block + log_block_size));
    RandomAccessIterator2 local_values_first2 = local_values_first1 + (1<<log_tile_size);
    
    // read in blocks
    // this causes unaligned loads to take place because start1 is usually unaligned.
    // We can do some fancy tricks to eliminate this unaligned load: somewhat better
    aligned_read<block_size>(local_keys_first1, local_values_first1, input1, input1val, start1, size1);
    
    // Read in other side
    aligned_read<block_size>(local_keys_first2, local_values_first2, input2, input2val, start2, size2);

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
    aligned_write<block_size>(input1, input1val, keys_result, values_result, (oddeven_blockid<<(log_num_merged_splitters_per_block + log_block_size)) + start1 + start2, size1 + size2);
  } // end for i
} // end merge_subblocks_binarysearch_kernel()

// merge_subblocks_binarysearch() merges each sub-block independently. As explained in find_splitter_ranks(), 
// the sub-blocks are defined by the ranks of the splitter elements d_rank1 and d_rank2 in the odd and even blocks resp.
// It can be easily shown that each sub-block cannot contain more than block_size elements of either the odd or even block.

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

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename RandomAccessIterator5,
         typename RandomAccessIterator6,
         typename RandomAccessIterator7,
         typename StrictWeakOrdering>
  void merge_subblocks_binarysearch(RandomAccessIterator1 keys_first,
                                    RandomAccessIterator2 values_first,
                                    unsigned int datasize, 
                                    RandomAccessIterator3 splitters_pos_first, 
                                    RandomAccessIterator4 ranks_first1,
                                    RandomAccessIterator5 ranks_first2, 
                                    RandomAccessIterator6 keys_result,
                                    RandomAccessIterator7 values_result, 
                                    unsigned int num_splitters, unsigned int log_tile_size, 
                                    unsigned int log_num_merged_splitters_per_block,
                                    unsigned int num_oddeven_tile_pairs,
                                    StrictWeakOrdering comp)
{
  typedef typename iterator_value<RandomAccessIterator6>::type KeyType;
  typedef typename iterator_value<RandomAccessIterator7>::type ValueType;

  unsigned int max_num_blocks = max_grid_size(1);

  unsigned int grid_size = min(num_oddeven_tile_pairs, max_num_blocks);

  const unsigned int log_block_size = merge_sort_dev_namespace::log_block_size<KeyType,ValueType>::result;

  // XXX WAR nvcc 3.0 unused variable warning
  unsigned int unused_variable_workaround = log_block_size;
  ++unused_variable_workaround;

  // Copy over the first merged splitter of each odd-even block pair to the output array
  copy_first_splitters<log_block_size><<<grid_size,1>>>(keys_first,
                                                        values_first,
                                                        splitters_pos_first, 
                                                        keys_result,
                                                        values_result,
                                                        log_num_merged_splitters_per_block,
                                                        num_oddeven_tile_pairs);
  synchronize_if_enabled("copy_first_splitters");

  const unsigned int block_size = merge_sort_dev_namespace::block_size<KeyType,ValueType>::result;

  max_num_blocks = max_grid_size(block_size);

  grid_size = min(num_splitters, max_num_blocks);

  merge_subblocks_binarysearch_kernel<block_size,log_block_size><<<grid_size, block_size, block_size*(2*sizeof(KeyType) + 2*sizeof(ValueType))>>>(
  	keys_first,
        values_first,
        datasize, 
  	ranks_first1,
        ranks_first2, 
  	log_tile_size,
        log_num_merged_splitters_per_block, 
        num_splitters,
  	keys_result,
        values_result,
        comp);
  synchronize_if_enabled("merge_subblocks_binarysearch_kernel");
}

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename StrictWeakOrdering>
  void merge(RandomAccessIterator1 keys_first,
             RandomAccessIterator2 values_first,
             size_t n,
             RandomAccessIterator3 keys_result,
             RandomAccessIterator4 values_result,
             size_t log_tile_size,
             size_t level,
             StrictWeakOrdering comp)
{
  typedef typename iterator_value<RandomAccessIterator3>::type KeyType;
  typedef typename iterator_value<RandomAccessIterator4>::type ValueType;

  size_t tile_size = 1<<log_tile_size;

  // assumption: num_tiles is even; tile_size is a power of 2
  size_t num_tiles = n / tile_size;
  size_t partial_tile_size = n % tile_size;
  if(partial_tile_size) ++num_tiles;

  // Compute the block_size based on the types to sort
  const unsigned int block_size = merge_sort_dev_namespace::block_size<KeyType,ValueType>::result;

  // Case (a): tile_size <= block_size
  if(tile_size <= block_size)
  {
    // Two or more tiles can fully fit into shared memory, and can be merged by one thread block.
    // In particular, we load (2*block_size) elements into shared memory, 
    //        and merge all the contained tile pairs using one thread block.   
    // We use (2*block_size) threads/thread block and grid_size * tile_size/(2*block_size) thread blocks.
    unsigned int tiles_per_block = (2*block_size) / tile_size;
    unsigned int partial_block_size = num_tiles % tiles_per_block;
    unsigned int number_of_tiles_in_last_block = partial_block_size ? partial_block_size : tiles_per_block;
    unsigned int num_blocks = num_tiles / tiles_per_block;
    if(partial_block_size) ++num_blocks;

    // compute the maximum number of blocks we can launch on this arch
    const unsigned int max_num_blocks = max_grid_size(2 * block_size);
    unsigned int grid_size = min(num_blocks, max_num_blocks);

    // figure out the size & index of the last tile of the last block
    unsigned int size_of_last_tile = partial_tile_size ? partial_tile_size : tile_size;
    unsigned int index_of_last_tile_in_last_block = number_of_tiles_in_last_block - 1;

    merge_smalltiles_binarysearch<2*block_size><<<grid_size,(2*block_size)>>>(keys_first,
                                                                              values_first,
                                                                              n,
                                                                              num_blocks - 1,
                                                                              index_of_last_tile_in_last_block,
                                                                              size_of_last_tile,
                                                                              keys_result,
                                                                              values_result,
                                                                              log_tile_size,
                                                                              comp);
    synchronize_if_enabled("merge_smalltiles_binarysearch");

    return;
  } // end if

  // Case (b) tile_size >= block_size

  // compute the maximum number of blocks we can launch on this arch
  const unsigned int max_num_blocks = max_grid_size(block_size);

  // Step 1 of the recursive case: extract one splitter per block_size entries in each odd-even block pair.
  // Store the splitter keys into splitters[level], and the array index in keys_first of the splitters
  // chosen into splitters_pos[level]
  size_t num_splitters = n / block_size;
  if(n % block_size) ++num_splitters;

  unsigned int grid_size = min<size_t>(num_splitters, max_num_blocks);

  using namespace thrust::detail;

  typedef thrust::detail::cuda_device_space_tag space;

  thrust::detail::uninitialized_array<KeyType,      space>      splitters(num_splitters);
  thrust::detail::uninitialized_array<unsigned int, space>      splitters_pos(num_splitters);
  thrust::detail::uninitialized_array<KeyType,      space>      merged_splitters(num_splitters);
  thrust::detail::uninitialized_array<unsigned int, space>      merged_splitters_pos(num_splitters);

  extract_splitters<<<grid_size, block_size>>>(keys_first, n, splitters.begin(), splitters_pos.begin());
  synchronize_if_enabled("extract_splitters");

  // compute the log base 2 of the block_size
  const unsigned int log_block_size = merge_sort_dev_namespace::log_block_size<KeyType,ValueType>::result;

  // Step 2 of the recursive case: merge these elements using merge
  // We need to merge num_splitters elements, each new "block" is the set of
  // splitters for each original block.
  size_t log_num_splitters_per_block = log_tile_size - log_block_size;
  merge(splitters.begin(),
        splitters_pos.begin(),
        num_splitters,
        merged_splitters.begin(),
        merged_splitters_pos.begin(),
        log_num_splitters_per_block,
        level + 1,
        comp);
  // free this memory before recursion
  destroy(splitters);

  // Step 3 of the recursive case: Find the ranks of each splitter in the respective two blocks.
  // Store the results into rank1[level] and rank2[level] for the even and odd block respectively.
  // rank1[level] and rank2[level] define the sub-block splits:
  //      Sub-block 0: Elements with indices less than rank1[0] in the odd block less than rank2[0] in the even
  //      Sub-block 1: Indices between rank1[0] and rank1[1] in the odd block and
  //                           between rank2[0] and rank2[1] in the even block
  //      ... and so on.
  size_t log_num_merged_splitters_per_block = log_num_splitters_per_block + 1;

  size_t num_blocks = num_splitters / block_size;
  if(num_splitters % block_size) ++num_blocks;

  grid_size = min<size_t>(num_blocks, max_num_blocks);

  // reuse the splitters_pos storage for rank1
  thrust::detail::uninitialized_array<unsigned int, space> &rank1 = splitters_pos;
  thrust::detail::uninitialized_array<unsigned int, space> rank2(num_splitters);

  find_splitter_ranks<block_size, log_block_size>
    <<<grid_size,block_size>>>
      (merged_splitters.begin(),
       merged_splitters_pos.begin(),
       rank1.begin(),
       rank2.begin(),
       keys_first,
       n,
       num_splitters,
       log_tile_size,
       log_num_merged_splitters_per_block,
       comp);
  synchronize_if_enabled("find_splitter_ranks");

  // Step 4 of the recursive case: merge each sub-block independently in parallel.
  merge_subblocks_binarysearch(keys_first,
                               values_first,
                               n,
                               merged_splitters_pos.begin(),
                               rank1.begin(),
                               rank2.begin(),
                               keys_result,
                               values_result,
                               num_splitters,
                               log_tile_size,
                               log_num_merged_splitters_per_block,
                               num_tiles / 2,
                               comp);
}

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename StrictWeakOrdering>
  void merge(RandomAccessIterator1 keys_first,
             RandomAccessIterator2 values_first,
             size_t n,
             RandomAccessIterator3 keys_result,
             RandomAccessIterator4 values_result,
             size_t block_size,
             StrictWeakOrdering comp)
{
  unsigned int log_block_size = (unsigned int)logb((float)block_size);
  unsigned int num_blocks = (n%block_size)?((n/block_size)+1):(n/block_size);

  merge(keys_first,
        values_first,
        (num_blocks%2)?((num_blocks-1)*block_size):n,
        keys_result,
        values_result,
        log_block_size,
        0,
        comp);

  if(num_blocks%2)
  {
    thrust::copy(keys_first + (num_blocks-1)*block_size,
                 keys_first + n,
                 keys_result + (num_blocks-1)*block_size);
    
    thrust::copy(values_first + (num_blocks-1)*block_size,
                 values_first + n,
                 values_result + (num_blocks-1)*block_size);
  }
}


} // end merge_sort_dev_namespace



template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
void stable_merge_sort(RandomAccessIterator first,
                       RandomAccessIterator last,
                       StrictWeakOrdering comp)
{
    // XXX it's potentially unsafe to pass the same array for keys & values
    //     implement a legit merge_sort_dev function later
    thrust::detail::backend::cuda::detail::stable_merge_sort_by_key(first, last, first, comp);
}


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void stable_merge_sort_by_key(RandomAccessIterator1 keys_first,
                                RandomAccessIterator1 keys_last,
                                RandomAccessIterator2 values_first,
                                StrictWeakOrdering comp)
{
  typedef typename thrust::iterator_traits<RandomAccessIterator1>::value_type KeyType;
  typedef typename thrust::iterator_traits<RandomAccessIterator2>::value_type ValueType;

  const size_t n = keys_last - keys_first;

  // don't launch an empty kernel
  if(n == 0) return;

  // compute the block_size based on the types we're sorting
  const unsigned int block_size = merge_sort_dev_namespace::block_size<KeyType,ValueType>::result;

  // compute the maximum number of blocks we can launch on this arch
  const unsigned int max_num_blocks = merge_sort_dev_namespace::max_grid_size(block_size);

  // first, sort within each block
  size_t num_blocks = n / block_size;
  if(n % block_size) ++num_blocks;

  size_t grid_size = merge_sort_dev_namespace::min<size_t>(num_blocks, max_num_blocks);

  // do an odd-even sort per block of data
  merge_sort_dev_namespace::stable_odd_even_block_sort_kernel<block_size><<<grid_size, block_size>>>(keys_first, values_first, comp, n);
  synchronize_if_enabled("stable_odd_even_block_sort_kernel");

  // allocate scratch space
  using namespace thrust::detail;
  thrust::detail::uninitialized_array<KeyType,   cuda_device_space_tag> temp_keys(n);
  thrust::detail::uninitialized_array<ValueType, cuda_device_space_tag> temp_vals(n);

  // give iterators simpler names
  RandomAccessIterator1 keys0 = keys_first;
  RandomAccessIterator2 vals0 = values_first;
  typename thrust::detail::uninitialized_array<KeyType,   cuda_device_space_tag>::iterator keys1 = temp_keys.begin();
  typename thrust::detail::uninitialized_array<ValueType, cuda_device_space_tag>::iterator vals1 = temp_vals.begin();

  // The log(n) iterations start here. Each call to 'merge' merges an odd-even pair of tiles
  // Currently uses additional arrays for sorted outputs.
  unsigned int iterations = 0;
  for(unsigned int tile_size = block_size;
      tile_size < n;
      tile_size *= 2)
  {
    if (iterations % 2)
      merge_sort_dev_namespace::merge(keys1, vals1, n, keys0, vals0, tile_size, comp);
    else
      merge_sort_dev_namespace::merge(keys0, vals0, n, keys1, vals1, tile_size, comp);
    ++iterations;
  }

  // this is to make sure that our data is finally in the data and keys arrays
  // and not in the temporary arrays
  if(iterations % 2)
  {
    thrust::copy(vals1, vals1 + n, vals0);
    thrust::copy(keys1, keys1 + n, keys0);
  }
} // end stable_merge_sort_by_key()

} // end namespace detail
} // end namespace cuda
} // end namespace backend
} // end namespace detail
} // end namespace thrust


__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

