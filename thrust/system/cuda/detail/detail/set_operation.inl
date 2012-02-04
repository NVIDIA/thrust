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

#include <thrust/scan.h>
#include <thrust/detail/minmax.h>

#include <thrust/system/cuda/detail/arch.h>
#include <thrust/system/cuda/detail/extern_shared_ptr.h>
#include <thrust/system/cuda/detail/block/copy.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/system/cuda/detail/detail/launch_closure.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
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


__host__ __device__
inline unsigned int align_size_to_int(unsigned int num_bytes)
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
__host__ __device__ __thrust_forceinline__
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
         typename Size,
         typename Context>
struct set_operation_closure
{
  const RandomAccessIterator1 first1;
  const RandomAccessIterator1 last1;
  const RandomAccessIterator2 first2;
  const RandomAccessIterator2 last2;
  RandomAccessIterator3 splitter_ranks1;
  RandomAccessIterator4 splitter_ranks2;
  const RandomAccessIterator5 result;
  RandomAccessIterator6 result_partition_sizes;
  StrictWeakOrdering comp;
  BlockConvergentSetOperation set_operation;
  Size num_partitions;
  Context context;

  typedef Context context_type;

  set_operation_closure(const RandomAccessIterator1 first1, 
                        const RandomAccessIterator1 last1,
                        const RandomAccessIterator2 first2,
                        const RandomAccessIterator2 last2,
                        RandomAccessIterator3 splitter_ranks1,
                        RandomAccessIterator4 splitter_ranks2,
                        const RandomAccessIterator5 result,
                        RandomAccessIterator6 result_partition_sizes,
                        StrictWeakOrdering comp,
                        BlockConvergentSetOperation set_operation,
                        Size num_partitions,
                        Context context = Context())
    : first1(first1), last1(last1), first2(first2), last2(last2),
      splitter_ranks1(splitter_ranks1), splitter_ranks2(splitter_ranks2),
      result(result), result_partition_sizes(result_partition_sizes), comp(comp),
      set_operation(set_operation), num_partitions(num_partitions), context(context)
  {}

  __device__ __thrust_forceinline__
  void operator()(void)
  {
    typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type1;
    typedef typename thrust::iterator_value<RandomAccessIterator2>::type value_type2;
  
    // allocate shared storage
    const unsigned int array_size1 = align_array_size_to_int<value_type1>(context.block_dimension());
    const unsigned int array_size2 = align_array_size_to_int<value_type2>(context.block_dimension());
    const unsigned int array_size3 = align_size_to_int(set_operation.get_temporary_array_size_in_number_of_bytes(context.block_dimension()));
    int *_shared1  = extern_shared_ptr<int>();
    int *_shared2  = _shared1 + array_size1;
    int *_shared3  = _shared2 + array_size2;
    int *_shared4  = _shared3 + array_size3;
  
    value_type1 *s_input1  = reinterpret_cast<value_type1*>(_shared1);
    value_type2 *s_input2  = reinterpret_cast<value_type2*>(_shared2);
    void        *s_scratch = reinterpret_cast<void*>(_shared3);
    value_type1 *s_result  = reinterpret_cast<value_type1*>(_shared4);
  
    // advance per-partition iterators
    splitter_ranks1        += context.block_index();
    splitter_ranks2        += context.block_index();
    result_partition_sizes += context.block_index();
  
    for(Size partition_idx     = context.block_index();
        partition_idx          < num_partitions;
        partition_idx          += context.grid_dimension(),
        splitter_ranks1        += context.grid_dimension(),
        splitter_ranks2        += context.grid_dimension(),
        result_partition_sizes += context.grid_dimension())
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
  
        input_end1 = first1 + *rank1;
        input_end2 = first2 + *rank2;
      }
  
      // find the beginning of the input and output if this is not the first partition
      if(partition_idx != 0)
      {
        typedef typename thrust::iterator_difference<RandomAccessIterator1>::type difference1;
        typedef typename thrust::iterator_difference<RandomAccessIterator2>::type difference2;
        RandomAccessIterator3 rank1 = splitter_ranks1 - 1;
        RandomAccessIterator4 rank2 = splitter_ranks2 - 1;
  
        difference1 size_of_preceding_input1 = *rank1;
        difference2 size_of_preceding_input2 = *rank2;
  
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
      difference1 s_input1_size = thrust::min<difference1>(context.block_dimension(), input_end1 - input_begin1);
      block::copy(context, input_begin1, input_begin1 + s_input1_size, s_input1);
  
      // load the second segment
      difference2 s_input2_size = thrust::min<difference2>(context.block_dimension(), input_end2 - input_begin2);
      block::copy(context, input_begin2, input_begin2 + s_input2_size, s_input2);
  
      context.barrier();
  
      s_result_end = set_operation(context,
                                   s_input1, s_input1 + s_input1_size,
                                   s_input2, s_input2 + s_input2_size,
                                   s_scratch,
                                   s_result,
                                   comp);
  
      context.barrier();
  
      // store to gmem
      block::copy(context, s_result, s_result_end, output_begin);
  
      // store size of the result
      if(context.thread_index() == 0)
      {
        *result_partition_sizes = s_result_end - s_result;
      } // end if
    } // end for partition
  }
}; // end set_operation_closure


template<typename BlockConvergentSetOperation,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename RandomAccessIterator5,
         typename Size,
         typename Context>
struct grouped_gather_closure
{
  const RandomAccessIterator1 result;
  const RandomAccessIterator2 first;
  RandomAccessIterator3 splitter_ranks1;
  RandomAccessIterator4 splitter_ranks2;
  RandomAccessIterator5 size_of_result_before_and_including_each_partition;
  Size num_partitions;
  Context context;

  typedef Context context_type;

  grouped_gather_closure(const RandomAccessIterator1 result,
                         const RandomAccessIterator2 first,
                         RandomAccessIterator3 splitter_ranks1,
                         RandomAccessIterator4 splitter_ranks2,
                         RandomAccessIterator5 size_of_result_before_and_including_each_partition,
                         Size num_partitions,
                         Context context = Context())
    : result(result), first(first), splitter_ranks1(splitter_ranks1), splitter_ranks2(splitter_ranks2),
      size_of_result_before_and_including_each_partition(size_of_result_before_and_including_each_partition),
      num_partitions(num_partitions), context(context) 
  {}

  __device__ __thrust_forceinline__
  void operator()(void)
  {
    using namespace thrust::detail::backend;
  
    // advance iterators
    splitter_ranks1                                     += context.block_index();
    splitter_ranks2                                     += context.block_index();
    size_of_result_before_and_including_each_partition  += context.block_index();
  
    for(Size partition_idx                                 = context.block_index();
        partition_idx                                      < num_partitions;
        partition_idx                                      += context.grid_dimension(),
        splitter_ranks1                                    += context.grid_dimension(),
        splitter_ranks2                                    += context.grid_dimension(),
        size_of_result_before_and_including_each_partition += context.grid_dimension())
    {
      RandomAccessIterator1 output_begin = result;
      RandomAccessIterator2 input_begin = first;
  
      // find the location of the input and output if this is not the first partition
      typename thrust::iterator_value<RandomAccessIterator4>::type partition_size = *size_of_result_before_and_including_each_partition;

      if(partition_idx != 0)
      {
        // advance iterators to the beginning of this partition's input
        RandomAccessIterator3 rank1 = splitter_ranks1;
        --rank1;
        RandomAccessIterator4 rank2 = splitter_ranks2;
        --rank2;
  
        typedef typename thrust::iterator_difference<RandomAccessIterator2>::type difference;
  
        difference size_of_preceding_input1 = *rank1;
        difference size_of_preceding_input2 = *rank2;
  
        input_begin +=
          BlockConvergentSetOperation::get_max_size_of_result_in_number_of_elements(size_of_preceding_input1,size_of_preceding_input2);
  
        // subtract away the previous element of size_of_result_preceding_each_partition
        // resulting in this size of this partition
        --size_of_result_before_and_including_each_partition;
        typename thrust::iterator_value<RandomAccessIterator4>::type beginning_of_output = *size_of_result_before_and_including_each_partition;
  
        output_begin += beginning_of_output;
        partition_size -= *size_of_result_before_and_including_each_partition;
      } // end if
  
      block::copy(context, input_begin, input_begin + partition_size, output_begin);
    } // end for partition
  }
}; // end grouped_gather_closure

} // end namespace set_operation_detail


template<typename Tag,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename Compare,
         typename SplittingFunction,
         typename BlockConvergentSetOperation>
  RandomAccessIterator3 set_operation(Tag,
                                      RandomAccessIterator1 first1,
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
  
  typedef detail::blocked_thread_array Context;

  typedef set_operation_closure<RandomAccessIterator1,
                                RandomAccessIterator2,
                                typename thrust::detail::temporary_array<difference1,Tag>::iterator,
                                typename thrust::detail::temporary_array<difference2,Tag>::iterator,
                                typename thrust::detail::temporary_array<value_type1,Tag>::iterator,
                                typename thrust::detail::temporary_array<difference1,Tag>::iterator,
                                Compare,
                                BlockConvergentSetOperation,
                                size_t,
                                Context> SetOperationClosure;

  arch::function_attributes_t attributes = detail::closure_attributes<SetOperationClosure>();
  arch::device_properties_t   properties = arch::device_properties();

  // prefer large blocks to keep the partitions as large as possible
  const size_t block_size =
    arch::max_blocksize_subject_to_smem_usage(properties, attributes,
                                              get_set_operation_kernel_per_block_dynamic_smem_usage<
                                                RandomAccessIterator1,
                                                RandomAccessIterator2,
                                                BlockConvergentSetOperation>);

  const size_t partition_size = block_size;
  const difference1 num_elements1 = last1 - first1;
  const difference2 num_elements2 = last2 - first2;

  const difference1 num_partitions1 = ceil_div(num_elements1, partition_size);
  const difference2 num_splitters_from_range1 = (num_partitions1 == 0) ? 0 : num_partitions1 - 1;

  const difference2 num_partitions2 = ceil_div(num_elements2, partition_size);
  const difference2 num_splitters_from_range2 = (num_partitions2 == 0) ? 0 : num_partitions2 - 1;

  const size_t num_merged_partitions = num_splitters_from_range1 + num_splitters_from_range2 + 1;

  // allocate storage for splitter ranks
  thrust::detail::temporary_array<difference1, Tag> splitter_ranks1(num_splitters_from_range1 + num_splitters_from_range2);
  thrust::detail::temporary_array<difference2, Tag> splitter_ranks2(num_splitters_from_range1 + num_splitters_from_range2);

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
  thrust::detail::temporary_array<difference1, Tag> result_partition_sizes(num_merged_partitions);

  // allocate storage to store the largest possible result
  // XXX if the size of the result is known a priori (i.e., first == second), we don't need this temporary
  temporary_array<value_type1, Tag>
    temporary_results(set_op.get_max_size_of_result_in_number_of_elements(num_elements1, num_elements2));

  // maximize the number of blocks we can launch
  const size_t max_blocks = properties.maxGridSize[0];
  const size_t num_blocks = thrust::min(num_merged_partitions, max_blocks);
  const size_t dynamic_smem_size = get_set_operation_kernel_per_block_dynamic_smem_usage<RandomAccessIterator1,RandomAccessIterator2,BlockConvergentSetOperation>(block_size);

  detail::launch_closure
    (SetOperationClosure(first1, last1,
                         first2, last2,
                         splitter_ranks1.begin(),
                         splitter_ranks2.begin(),
                         temporary_results.begin(),
                         result_partition_sizes.begin(),
                         comp,
                         set_op,
                         num_merged_partitions),
     num_blocks, block_size, dynamic_smem_size);

  // inclusive scan the element counts to get the number of elements occurring before and including each partition
  thrust::inclusive_scan(result_partition_sizes.begin(),
                         result_partition_sizes.end(),
                         result_partition_sizes.begin());

  // gather from temporary_results to result
  // no real need to choose a new config for this launch
  typedef grouped_gather_closure<
    BlockConvergentSetOperation,
    RandomAccessIterator3,
    typename temporary_array<value_type1, Tag>::iterator,
    typename temporary_array<difference1, Tag>::iterator,
    typename temporary_array<difference2, Tag>::iterator,
    typename temporary_array<difference1, Tag>::iterator,
    size_t,
    Context
  > GroupedGatherClosure;

  detail::launch_closure
    (GroupedGatherClosure(result,
                          temporary_results.begin(),
                          splitter_ranks1.begin(),
                          splitter_ranks2.begin(),
                          result_partition_sizes.begin(),
                          num_merged_partitions),
     num_blocks, block_size);

  return result + result_partition_sizes[num_merged_partitions - 1];
} // end set_operation()

} // end namespace detail
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

