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

#include <thrust/detail/config.h>

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#include <thrust/detail/backend/dereference.h>
#include <thrust/detail/backend/cuda/synchronize.h>
#include <thrust/detail/backend/cuda/arch.h>
#include <thrust/detail/backend/cuda/extern_shared_ptr.h>
#include <thrust/detail/backend/cuda/block/copy.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/detail/uninitialized_array.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace cuda
{
namespace detail
{

namespace set_operation_detail
{


template<typename T1, typename T2>
T1 ceil_div(T1 up, T2 down)
{
  T1 div = up / down;
  T1 rem = up % down;
  return (rem != 0) ? div + 1 : div;
}


inline __host__ __device__
unsigned int align_size_to_int(unsigned int num_bytes)
{
  return (num_bytes / sizeof(int)) + ((num_bytes % sizeof(int)) ? 1 : 0);
}


/*! \param num_elements The minimum number of array elements of type \p T desired.
 *  \return The minimum number of elements >= \p num_elements such that the size
 *          of an array of elements of type \p T is aligned to \c int.
 *  \note Another way to interpret this function is that it returns the minimum number
 *        of \c ints that would accomodate a contiguous array of \p num_elements \p Ts.
 */
template<typename T>
inline __host__ __device__
unsigned int align_array_size_to_int(unsigned int num_elements)
{
  unsigned int num_bytes = num_elements * sizeof(T);
  return align_size_to_int(num_bytes);
}


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename BlockConvergentSetOperation>
unsigned int get_set_operation_kernel_per_block_dynamic_smem_usage(unsigned int block_size)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type1;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type value_type2;

  // set_operation_kernel allocates memory aligned to int
  const unsigned int array_size1 = align_array_size_to_int<value_type1>(block_size);
  const unsigned int array_size2 = align_array_size_to_int<value_type2>(block_size);
  const unsigned int array_size3 = align_size_to_int(BlockConvergentSetOperation::get_temporary_array_size_in_number_of_bytes(block_size));
  const unsigned int array_size4 = align_size_to_int(sizeof(value_type1) * BlockConvergentSetOperation::get_max_size_of_result_in_number_of_elements(block_size,block_size));

  return sizeof(int) * (array_size1 + array_size2 + array_size3 + array_size4);
} // end get_set_operation_kernel_per_block_dynamic_smem_usage()


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename RandomAccessIterator5,
         typename RandomAccessIterator6,
         typename StrictWeakOrdering,
         typename BlockConvergentSetOperation,
         typename Size>
__global__ void set_operation_kernel(const RandomAccessIterator1 first1, 
                                     const RandomAccessIterator1 last1,
                                     const RandomAccessIterator2 first2,
                                     const RandomAccessIterator2 last2,
                                     RandomAccessIterator3 splitter_ranks1,
                                     RandomAccessIterator4 splitter_ranks2,
                                     const RandomAccessIterator5 result,
                                     RandomAccessIterator6 result_partition_sizes,
                                     StrictWeakOrdering comp,
                                     BlockConvergentSetOperation set_operation,
                                     Size num_partitions)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type1;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type value_type2;

  // allocate shared storage
  const unsigned int array_size1 = align_array_size_to_int<value_type1>(blockDim.x);
  const unsigned int array_size2 = align_array_size_to_int<value_type2>(blockDim.x);
  const unsigned int array_size3 = align_size_to_int(set_operation.get_temporary_array_size_in_number_of_bytes(blockDim.x));
  int *_shared1  = extern_shared_ptr<int>();
  int *_shared2  = _shared1 + array_size1;
  int *_shared3  = _shared2 + array_size2;
  int *_shared4  = _shared3 + array_size3;

  value_type1 *s_input1  = reinterpret_cast<value_type1*>(_shared1);
  value_type2 *s_input2  = reinterpret_cast<value_type2*>(_shared2);
  void        *s_scratch = reinterpret_cast<void*>(_shared3);
  value_type1 *s_result  = reinterpret_cast<value_type1*>(_shared4);

  // advance per-partition iterators
  splitter_ranks1        += blockIdx.x;
  splitter_ranks2        += blockIdx.x;
  result_partition_sizes += blockIdx.x;

  for(Size partition_idx     = blockIdx.x;
      partition_idx          < num_partitions;
      partition_idx          += gridDim.x,
      splitter_ranks1        += gridDim.x,
      splitter_ranks2        += gridDim.x,
      result_partition_sizes += gridDim.x)
  {
    RandomAccessIterator1 input_begin1 = first1;
    RandomAccessIterator1 input_end1   = last1;
    RandomAccessIterator2 input_begin2 = first2;
    RandomAccessIterator2 input_end2   = last2;

    RandomAccessIterator5 output_begin = result;

    // find the end of the input if this is not the last block
    if(partition_idx != num_partitions - 1)
    {
      RandomAccessIterator3 rank1 = splitter_ranks1;
      RandomAccessIterator4 rank2 = splitter_ranks2;

      input_end1 = first1 + dereference(rank1);
      input_end2 = first2 + dereference(rank2);
    }

    // find the beginning of the input and output if this is not the first partition
    if(partition_idx != 0)
    {
      typedef typename thrust::iterator_difference<RandomAccessIterator1>::type difference1;
      typedef typename thrust::iterator_difference<RandomAccessIterator2>::type difference2;
      RandomAccessIterator3 rank1 = splitter_ranks1;
      --rank1;
      RandomAccessIterator4 rank2 = splitter_ranks2;
      --rank2;

      difference1 size_of_preceding_input1 = dereference(rank1);
      difference2 size_of_preceding_input2 = dereference(rank2);

      // advance the input to point to the beginning
      input_begin1 += size_of_preceding_input1;
      input_begin2 += size_of_preceding_input2;

      // conservatively advance the result to point to the beginning of the output
      output_begin += set_operation.get_max_size_of_result_in_number_of_elements(size_of_preceding_input1,size_of_preceding_input2);
    }

    // the result is empty to begin with
    value_type1 *s_result_end = s_result;

    typedef typename thrust::iterator_difference<RandomAccessIterator1>::type difference1;

    typedef typename thrust::iterator_difference<RandomAccessIterator2>::type difference2;

    // load the first segment
    difference1 s_input1_size = thrust::min<difference1>(blockDim.x, input_end1 - input_begin1);
    block::copy(input_begin1, input_begin1 + s_input1_size, s_input1);

    // load the second segment
    difference2 s_input2_size = thrust::min<difference2>(blockDim.x, input_end2 - input_begin2);
    block::copy(input_begin2, input_begin2 + s_input2_size, s_input2);

    __syncthreads();

    s_result_end = set_operation(s_input1, s_input1 + s_input1_size,
                                 s_input2, s_input2 + s_input2_size,
                                 s_scratch,
                                 s_result,
                                 comp);

    __syncthreads();

    // store to gmem
    block::copy(s_result, s_result_end, output_begin);

    // store size of the result
    if(threadIdx.x == 0)
    {
      dereference(result_partition_sizes) = s_result_end - s_result;
    } // end if
  } // end for partition
} // end set_operation_kernel()


template<typename BlockConvergentSetOperation,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename RandomAccessIterator5,
         typename Size>
__global__
void grouped_gather(const RandomAccessIterator1 result,
                    const RandomAccessIterator2 first,
                    RandomAccessIterator3 splitter_ranks1,
                    RandomAccessIterator4 splitter_ranks2,
                    RandomAccessIterator5 size_of_result_before_and_including_each_partition,
                    Size num_partitions)
{
  using namespace thrust::detail::backend;

  // advance iterators
  splitter_ranks1                                     += blockIdx.x;
  splitter_ranks2                                     += blockIdx.x;
  size_of_result_before_and_including_each_partition  += blockIdx.x;

  for(Size partition_idx                                 = blockIdx.x;
      partition_idx                                      < num_partitions;
      partition_idx                                      += gridDim.x,
      splitter_ranks1                                    += gridDim.x,
      splitter_ranks2                                    += gridDim.x,
      size_of_result_before_and_including_each_partition += gridDim.x)
  {
    RandomAccessIterator1 output_begin = result;
    RandomAccessIterator2 input_begin = first;

    // find the location of the input and output if this is not the first partition
    typename thrust::iterator_value<RandomAccessIterator4>::type partition_size
      = dereference(size_of_result_before_and_including_each_partition);
    if(partition_idx != 0)
    {
      // advance iterators to the beginning of this partition's input
      RandomAccessIterator3 rank1 = splitter_ranks1;
      --rank1;
      RandomAccessIterator4 rank2 = splitter_ranks2;
      --rank2;

      typedef typename thrust::iterator_difference<RandomAccessIterator2>::type difference;

      difference size_of_preceding_input1 = dereference(rank1);
      difference size_of_preceding_input2 = dereference(rank2);

      input_begin +=
        BlockConvergentSetOperation::get_max_size_of_result_in_number_of_elements(size_of_preceding_input1,size_of_preceding_input2);

      // subtract away the previous element of size_of_result_preceding_each_partition
      // resulting in this size of this partition
      --size_of_result_before_and_including_each_partition;
      typename thrust::iterator_value<RandomAccessIterator4>::type beginning_of_output
        = dereference(size_of_result_before_and_including_each_partition);

      output_begin += beginning_of_output;
      partition_size -= dereference(size_of_result_before_and_including_each_partition);
    } // end if

    thrust::detail::backend::cuda::block::copy(input_begin, input_begin + partition_size, output_begin);
  } // end for partition
} // end grouped_gather


} // end namespace set_operation_detail


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename Compare,
         typename SplittingFunction,
         typename BlockConvergentSetOperation>
  RandomAccessIterator3 set_operation(RandomAccessIterator1 first1,
                                      RandomAccessIterator1 last1,
                                      RandomAccessIterator2 first2,
                                      RandomAccessIterator2 last2,
                                      RandomAccessIterator3 result,
                                      Compare comp,
                                      SplittingFunction split,
                                      BlockConvergentSetOperation set_op)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type      value_type1;
  typedef typename thrust::iterator_difference<RandomAccessIterator1>::type difference1;
  typedef typename thrust::iterator_difference<RandomAccessIterator2>::type difference2;

  using namespace set_operation_detail;

  // prefer large blocks to keep the partitions as large as possible
  const size_t block_size =
    arch::max_blocksize_subject_to_smem_usage(set_operation_kernel<
                                                RandomAccessIterator1,
                                                RandomAccessIterator2,
                                                typename thrust::detail::uninitialized_array<difference1,cuda_device_space_tag>::iterator,
                                                typename thrust::detail::uninitialized_array<difference2,cuda_device_space_tag>::iterator,
                                                typename thrust::detail::uninitialized_array<value_type1,cuda_device_space_tag>::iterator,
                                                typename thrust::detail::uninitialized_array<difference1,cuda_device_space_tag>::iterator,
                                                Compare,
                                                BlockConvergentSetOperation,
                                                size_t
                                              >,
                                              get_set_operation_kernel_per_block_dynamic_smem_usage<
                                                RandomAccessIterator1,
                                                RandomAccessIterator2,
                                                BlockConvergentSetOperation
                                              >);

  const size_t partition_size = block_size;
  const difference1 num_elements1 = last1 - first1;
  const difference2 num_elements2 = last2 - first2;

  const difference1 num_partitions1 = ceil_div(num_elements1, partition_size);
  const difference2 num_splitters_from_range1 = (num_partitions1 == 0) ? 0 : num_partitions1 - 1;

  const difference2 num_partitions2 = ceil_div(num_elements2, partition_size);
  const difference2 num_splitters_from_range2 = (num_partitions2 == 0) ? 0 : num_partitions2 - 1;

  const size_t num_merged_partitions = num_splitters_from_range1 + num_splitters_from_range2 + 1;

  // allocate storage for splitter ranks
  thrust::detail::uninitialized_array<difference1, cuda_device_space_tag> splitter_ranks1(num_splitters_from_range1 + num_splitters_from_range2);
  thrust::detail::uninitialized_array<difference2, cuda_device_space_tag> splitter_ranks2(num_splitters_from_range1 + num_splitters_from_range2);

  // select some splitters and find the rank of each splitter in the other range
  // XXX it's possible to fuse rank-finding with the kernel below
  //     this eliminates the temporary buffers splitter_ranks1 & splitter_ranks2
  //     but this spills to lmem and causes a 10x speeddown
  split(first1,last1,
        first2,last2,
        splitter_ranks1.begin(),
        splitter_ranks2.begin(),
        comp,
        partition_size,
        num_splitters_from_range1,
        num_splitters_from_range2);

  using namespace thrust::detail;

  // allocate storage to store each intersected partition's size
  thrust::detail::uninitialized_array<difference1, cuda_device_space_tag> result_partition_sizes(num_merged_partitions);

  // allocate storage to store the largest possible result
  // XXX if the size of the result is known a priori (i.e., first == second), we don't need this temporary
  thrust::detail::uninitialized_array<typename thrust::iterator_value<RandomAccessIterator1>::type, cuda_device_space_tag>
    temporary_results(set_op.get_max_size_of_result_in_number_of_elements(num_elements1, num_elements2));

  // maximize the number of blocks we can launch
  const size_t max_blocks = thrust::detail::backend::cuda::arch::device_properties().maxGridSize[0];
  const size_t num_blocks = thrust::min(num_merged_partitions, max_blocks);
  const size_t dynamic_smem_size = get_set_operation_kernel_per_block_dynamic_smem_usage<RandomAccessIterator1,RandomAccessIterator2,BlockConvergentSetOperation>(block_size);

  set_operation_kernel<<<num_blocks, static_cast<unsigned int>(block_size), static_cast<unsigned int>(dynamic_smem_size)>>>( 
  	first1, last1,
        first2, last2,
  	splitter_ranks1.begin(),
  	splitter_ranks2.begin(),
  	temporary_results.begin(), 
        result_partition_sizes.begin(),
  	comp,
        set_op,
        num_merged_partitions);
  synchronize_if_enabled("set_operation_kernel");

  // inclusive scan the element counts to get the number of elements occurring before and including each partition
  thrust::inclusive_scan(result_partition_sizes.begin(),
                         result_partition_sizes.end(),
                         result_partition_sizes.begin());

  // gather from temporary_results to result
  // no real need to choose a new config for this launch
  grouped_gather<BlockConvergentSetOperation>
    <<<num_blocks, static_cast<unsigned int>(block_size)>>>(
      result,
      temporary_results.begin(),
      splitter_ranks1.begin(),
      splitter_ranks2.begin(),
      result_partition_sizes.begin(),
      num_merged_partitions);
  synchronize_if_enabled("grouped_gather");

  return result + result_partition_sizes[num_merged_partitions - 1];
} // end set_operation()


} // end detail
} // end cuda
} // end backend
} // end detail
} // end thrust

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

