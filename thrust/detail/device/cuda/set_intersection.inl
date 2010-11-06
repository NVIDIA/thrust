/*
 *  Copyright 2008-2010 NVIDIA Corporation
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

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/device/dereference.h>
#include <thrust/extrema.h>
#include <thrust/detail/device/cuda/arch.h>
#include <thrust/detail/device/cuda/block/copy.h>
#include <thrust/detail/device/cuda/block/set_intersection.h>
#include <thrust/detail/device/cuda/synchronize.h>
#include <thrust/detail/device/cuda/arch.h>
#include <thrust/detail/device/cuda/extern_shared_ptr.h>
#include <thrust/detail/device/cuda/detail/get_set_operation_splitter_ranks.h>
#include <thrust/scan.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{

namespace set_intersection_detail
{

template<typename T1, typename T2>
T1 ceil_div(T1 up, T2 down)
{
  T1 div = up / down;
  T1 rem = up % down;
  return (rem != 0) ? div + 1 : div;
}

__host__ __device__
inline unsigned int align_size_to_int(unsigned int N)
{
  return (N / sizeof(int)) + ((N % sizeof(int)) ? 1 : 0);
}

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator5>
unsigned int get_set_intersection_kernel_per_block_dynamic_smem_usage(unsigned int block_size)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type1;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type value_type2;

  // set_intersection_kernel allocates memory aligned to int
  const unsigned int array_size1 = align_size_to_int(block_size * sizeof(value_type1));
  const unsigned int array_size2 = align_size_to_int(block_size * sizeof(value_type2));
  const unsigned int array_size3 = align_size_to_int(block_size * sizeof(int));
  const unsigned int array_size4 = align_size_to_int(block_size * sizeof(value_type1));

  return sizeof(int) * (array_size1 + array_size2 + array_size3 + array_size4);
} // end get_set_intersection_kernel_per_block_dynamic_smem_usage()

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2, 
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename RandomAccessIterator5,
         typename RandomAccessIterator6,
         typename StrictWeakOrdering,
         typename Size>
__global__ void set_intersection_kernel(const RandomAccessIterator1 first1, 
                                        const RandomAccessIterator1 last1,
                                        const RandomAccessIterator2 first2,
                                        const RandomAccessIterator2 last2,
                                        RandomAccessIterator3 splitter_ranks1,
                                        RandomAccessIterator4 splitter_ranks2,
                                        const RandomAccessIterator5 result,
                                        RandomAccessIterator6 result_partition_sizes,
                                        StrictWeakOrdering comp,
                                        Size num_partitions)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type1;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type value_type2;

  // allocate shared storage
  const unsigned int array_size1 = align_size_to_int(blockDim.x * sizeof(value_type1));
  const unsigned int array_size2 = align_size_to_int(blockDim.x * sizeof(value_type2));
  const unsigned int array_size3 = align_size_to_int(blockDim.x * sizeof(int));
  const unsigned int array_size4 = align_size_to_int(blockDim.x * sizeof(value_type1));
  int *_shared1  = extern_shared_ptr<int>();
  int *_shared2  = _shared1 + array_size1;
  int *s_scratch = _shared2 + array_size2;
  int *_shared3  = s_scratch + array_size3;

  value_type1 *s_input1 = reinterpret_cast<value_type1*>(_shared1);
  value_type2 *s_input2 = reinterpret_cast<value_type2*>(_shared2);
  value_type1 *s_result = reinterpret_cast<value_type1*>(_shared3);

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
    // the end of merged partition i is at splitter_ranks1[i] + splitter_ranks2[i]
    if(partition_idx != num_partitions - 1)
    {
      RandomAccessIterator3 rank1 = splitter_ranks1;
      RandomAccessIterator4 rank2 = splitter_ranks2;

      input_end1 = first1 + dereference(rank1);
      input_end2 = first2 + dereference(rank2);
    }

    // find the beginning of the input and output if this is not the first partition
    // merged partition i begins at splitter_ranks1[i-1]
    if(partition_idx != 0)
    {
      RandomAccessIterator3 rank1 = splitter_ranks1;
      --rank1;
      RandomAccessIterator4 rank2 = splitter_ranks2;
      --rank2;

      // advance the input to point to the beginning
      input_begin1 += dereference(rank1);
      input_begin2 += dereference(rank2);

      // advance the result to point to the beginning of the output
      output_begin += dereference(rank1);
    }

    // the result is empty to begin with
    value_type1 *s_result_end = s_result;

    if(input_begin1 < input_end1 && input_begin2 < input_end2)
    {
      typedef typename thrust::iterator_difference<RandomAccessIterator1>::type difference1;

      typedef typename thrust::iterator_difference<RandomAccessIterator2>::type difference2;

      // load the first segment
      difference1 s_input1_size = thrust::min<difference1>(blockDim.x, input_end1 - input_begin1);

      block::copy(input_begin1, input_begin1 + s_input1_size, s_input1);
      input_begin1 += s_input1_size;

      // load the second segment
      difference2 s_input2_size = thrust::min<difference2>(blockDim.x, input_end2 - input_begin2);

      block::copy(input_begin2, input_begin2 + s_input2_size, s_input2);
      input_begin2 += s_input2_size;

      __syncthreads();

      s_result_end = block::set_intersection(s_input1, s_input1 + s_input1_size,
                                             s_input2, s_input2 + s_input2_size,
                                             s_scratch,
                                             s_result,
                                             comp);

      __syncthreads();

      // store to gmem
      output_begin = block::copy(s_result, s_result_end, output_begin);
    } // end if

    // store size of the result
    if(threadIdx.x == 0)
    {
      dereference(result_partition_sizes) = s_result_end - s_result;
    } // end if
  } // end for partition
} // end set_intersection_kernel

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename Size>
__global__
void grouped_gather(const RandomAccessIterator1 result,
                    const RandomAccessIterator2 first,
                    RandomAccessIterator3 splitter_ranks1,
                    RandomAccessIterator4 size_of_result_before_and_including_each_partition,
                    Size num_partitions)
{
  using namespace thrust::detail::device;

  // advance iterators
  splitter_ranks1                                     += blockIdx.x;
  size_of_result_before_and_including_each_partition  += blockIdx.x;

  for(Size partition_idx                                 = blockIdx.x;
      partition_idx                                      < num_partitions;
      partition_idx                                      += gridDim.x,
      splitter_ranks1                                    += gridDim.x,
      size_of_result_before_and_including_each_partition += gridDim.x)
  {
    RandomAccessIterator1 output_begin = result;
    RandomAccessIterator2 input_begin = first;

    // find the location of the input and output if this is not the first partition
    typename thrust::iterator_value<RandomAccessIterator4>::type partition_size
      = dereference(size_of_result_before_and_including_each_partition);
    if(partition_idx != 0)
    {
      // advance iterator to the beginning of this partition's input
      RandomAccessIterator3 rank = splitter_ranks1;
      --rank;
      input_begin += dereference(rank);

      // subtract away the previous element of size_of_result_preceding_each_partition
      // resulting in this size of this partition
      --size_of_result_before_and_including_each_partition;
      typename thrust::iterator_value<RandomAccessIterator4>::type beginning_of_output
        = dereference(size_of_result_before_and_including_each_partition);

      output_begin += beginning_of_output;
      partition_size -= dereference(size_of_result_before_and_including_each_partition);
    } // end if

    thrust::detail::device::cuda::block::copy(input_begin, input_begin + partition_size, output_begin);
  } // end for partition
} // end grouped_gather


template<typename Compare>
  struct equivalence_from_compare
{
  equivalence_from_compare(Compare c)
    : comp(c) {}

  template<typename T1, typename T2>
  __host__ __device__
  bool operator()(T1 &lhs, T2 &rhs)
  {
    // two elements are equivalent if neither sorts before the other
    return !comp(lhs, rhs) && !comp(rhs, lhs);
  }
  
  Compare comp;
}; // end equivalence_from_compare


// this predicate tests two two-element tuples
// we first use a Compare for the first element
// if the first elements are equivalent, we use
// < for the second elements
// XXX set_intersection duplicates this
//     move it some place common
template<typename Compare>
  struct compare_first_less_second
{
  compare_first_less_second(Compare c)
    : comp(c) {}

  template<typename T1, typename T2>
  __host__ __device__
  bool operator()(T1 lhs, T2 rhs)
  {
    return comp(lhs.get<0>(), rhs.get<0>()) || (!comp(rhs.get<0>(), lhs.get<0>()) && lhs.get<1>() < rhs.get<1>());
  }

  Compare comp;
}; // end compare_first_less_second


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename Compare,
         typename Size1,
         typename Size2>
  void get_set_intersection_splitter_ranks(RandomAccessIterator1 first1,
                                           RandomAccessIterator1 last1,
                                           RandomAccessIterator2 first2,
                                           RandomAccessIterator2 last2,
                                           RandomAccessIterator3 splitter_ranks1,
                                           RandomAccessIterator4 splitter_ranks2,
                                           Compare comp,
                                           Size1 partition_size,
                                           Size2 num_splitters_from_each_range)
{
  using namespace thrust::detail;

  typedef typename thrust::iterator_difference<RandomAccessIterator1>::type difference1;
  typedef typename thrust::iterator_difference<RandomAccessIterator2>::type difference2;

  const difference1 num_elements1 = last1 - first1;
  const difference2 num_elements2 = last2 - first2;

  // enumerate each key within its sub-segment of equivalent keys
  // XXX replace these explicit ranges with fancy iterators
  raw_buffer<difference1, cuda_device_space_tag> key_ranks1(num_elements1);
  raw_buffer<difference2, cuda_device_space_tag> key_ranks2(num_elements2);

  thrust::exclusive_scan_by_key(first1, last1,
                                thrust::make_constant_iterator<difference1>(1),
                                key_ranks1.begin(),
                                difference1(0),
                                equivalence_from_compare<Compare>(comp));

  thrust::exclusive_scan_by_key(first2, last2,
                                thrust::make_constant_iterator<difference2>(1),
                                key_ranks2.begin(),
                                difference2(0),
                                equivalence_from_compare<Compare>(comp));

  // zip up the keys with their ranks to disambiguate repeated elements during rank-finding
  typedef typename raw_buffer<difference1, cuda_device_space_tag>::iterator RankIterator1;
  typedef typename raw_buffer<difference2, cuda_device_space_tag>::iterator RankIterator2;
  typedef thrust::tuple<RandomAccessIterator1,RankIterator1> iterator_tuple1;
  typedef thrust::tuple<RandomAccessIterator2,RankIterator2> iterator_tuple2;
  typedef thrust::zip_iterator<iterator_tuple1> iterator_and_rank1;
  typedef thrust::zip_iterator<iterator_tuple2> iterator_and_rank2;

  iterator_and_rank1 first_and_rank1 =
    thrust::make_zip_iterator(thrust::make_tuple(first1, key_ranks1.begin()));
  iterator_and_rank1 last_and_rank1 = first_and_rank1 + num_elements1;

  iterator_and_rank2 first_and_rank2 =
    thrust::make_zip_iterator(thrust::make_tuple(first2, key_ranks2.begin()));
  iterator_and_rank2 last_and_rank2 = first_and_rank2 + num_elements2;

  // take into account the tuples when comparing
  typedef compare_first_less_second<Compare> splitter_compare;

  using namespace thrust::detail::device::cuda::detail;
  return get_set_operation_splitter_ranks(first_and_rank1, last_and_rank1,
                                          first_and_rank2, last_and_rank2,
                                          splitter_ranks1,
                                          splitter_ranks2,
                                          splitter_compare(comp),
                                          partition_size,
                                          num_splitters_from_each_range);
} // end get_set_intersection_splitter_ranks()


} // end namespace set_intersection_detail


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2, 
	 typename RandomAccessIterator3,
         typename Compare>
RandomAccessIterator3 set_intersection(RandomAccessIterator1 first1,
                                       RandomAccessIterator1 last1,
                                       RandomAccessIterator2 first2,
                                       RandomAccessIterator2 last2,
                                       RandomAccessIterator3 result,
                                       Compare comp)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type      value_type1;
  typedef typename thrust::iterator_difference<RandomAccessIterator1>::type difference1;
  typedef typename thrust::iterator_difference<RandomAccessIterator2>::type difference2;

  const difference1 num_elements1 = last1 - first1;
  const difference2 num_elements2 = last2 - first2;

  // check for trivial problem
  if(num_elements1 == 0 || num_elements2 == 0)
    return result;

  using namespace set_intersection_detail;
  using namespace thrust::detail;
  using namespace thrust::detail::device::cuda::detail;

  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type;
  
  // prefer large blocks to keep the partitions as large as possible
  const size_t block_size =
    arch::max_blocksize_subject_to_smem_usage(set_intersection_kernel<
                                                RandomAccessIterator1,
                                                RandomAccessIterator2,
                                                typename raw_buffer<difference1,cuda_device_space_tag>::iterator,
                                                typename raw_buffer<difference2,cuda_device_space_tag>::iterator,
                                                typename raw_buffer<value_type1,cuda_device_space_tag>::iterator,
                                                typename raw_buffer<difference1,cuda_device_space_tag>::iterator,
                                                Compare,
                                                size_t
                                              >,
                                              get_set_intersection_kernel_per_block_dynamic_smem_usage<
                                                RandomAccessIterator1,
                                                RandomAccessIterator2,
                                                typename raw_buffer<value_type1,cuda_device_space_tag>::iterator
                                              >);

  const size_t partition_size = block_size;
  const difference1 num_partitions = ceil_div(num_elements1, partition_size);
  const difference1 num_splitters_from_each_range  = num_partitions - 1;
  const size_t num_merged_partitions = 2 * num_splitters_from_each_range + 1;

  // allocate storage for splitter ranks
  raw_buffer<difference1, cuda_device_space_tag> splitter_ranks1(2 * num_splitters_from_each_range);
  raw_buffer<difference2, cuda_device_space_tag> splitter_ranks2(2 * num_splitters_from_each_range);

  // select some splitters and find the rank of each splitter in the other range
  // XXX it's possible to fuse rank-finding with the merge_kernel below
  //     this eliminates the temporary buffers splitter_ranks1 & splitter_ranks2
  //     but this spills to lmem and causes a 10x speeddown
  get_set_intersection_splitter_ranks(first1,last1,
                                      first2,last2,
                                      splitter_ranks1.begin(),
                                      splitter_ranks2.begin(),
                                      comp,
                                      partition_size,
                                      num_splitters_from_each_range);

  // allocate storage to store each intersected partition's size
  raw_buffer<difference1, cuda_device_space_tag> result_partition_sizes(num_merged_partitions);

  // allocate storage to store the largest possible intersection
  raw_buffer<typename thrust::iterator_value<RandomAccessIterator1>::type, cuda_device_space_tag> temporary_results(num_elements1);

  // maximize the number of blocks we can launch
  const size_t max_blocks = thrust::detail::device::cuda::arch::max_grid_dimensions().x;
  const size_t num_blocks = thrust::min(num_merged_partitions, max_blocks);
  const size_t dynamic_smem_size = get_set_intersection_kernel_per_block_dynamic_smem_usage<RandomAccessIterator1,RandomAccessIterator2,RandomAccessIterator3>(block_size);

  set_intersection_detail::set_intersection_kernel<<<num_blocks, static_cast<unsigned int>(block_size), static_cast<unsigned int>(dynamic_smem_size)>>>( 
  	first1, last1,
        first2, last2,
  	splitter_ranks1.begin(),
  	splitter_ranks2.begin(),
  	temporary_results.begin(), 
        result_partition_sizes.begin(),
  	comp,
        num_merged_partitions);
  synchronize_if_enabled("set_intersection_kernel");

  // inclusive scan the element counts to get the number of elements occurring before and including each partition
  thrust::inclusive_scan(result_partition_sizes.begin(),
                         result_partition_sizes.end(),
                         result_partition_sizes.begin());

  // gather from temporary_results to result
  // XXX use a different heuristic for this kernel config
  grouped_gather<<<num_blocks, static_cast<unsigned int>(block_size)>>>(
                 result,
                 temporary_results.begin(),
                 splitter_ranks1.begin(),
                 result_partition_sizes.begin(),
                 num_merged_partitions);
  synchronize_if_enabled("grouped_gather");

  return result + result_partition_sizes[num_merged_partitions - 1];
} // end merge

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

