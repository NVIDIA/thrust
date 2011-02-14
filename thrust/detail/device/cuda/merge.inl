/*
 *  Copyright 2008-2011 NVIDIA Corporation
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
#include <thrust/detail/device/cuda/block/merge.h>
#include <thrust/detail/device/cuda/synchronize.h>
#include <thrust/detail/device/cuda/arch.h>
#include <thrust/detail/device/cuda/extern_shared_ptr.h>
#include <thrust/detail/device/cuda/detail/get_set_operation_splitter_ranks.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/detail/internal_functional.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{

namespace merge_detail
{

template<typename T1, typename T2>
T1 ceil_div(T1 up, T2 down)
{
  T1 div = up / down;
  T1 rem = up % down;
  return (rem != 0) ? div + 1 : div;
}

template<unsigned int N>
  struct static_align_size_to_int
{
  static const unsigned int value = (N / sizeof(int)) + ((N % sizeof(int)) ? 1 : 0);
};

__host__ __device__
inline unsigned int align_size_to_int(unsigned int N)
{
  return (N / sizeof(int)) + ((N % sizeof(int)) ? 1 : 0);
}

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator5>
unsigned int get_merge_kernel_per_block_dynamic_smem_usage(unsigned int block_size)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type1;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type value_type2;
  typedef typename thrust::iterator_value<RandomAccessIterator5>::type value_type5;

  // merge_kernel allocates memory aligned to int
  const unsigned int array_size1 = align_size_to_int(block_size * sizeof(value_type1));
  const unsigned int array_size2 = align_size_to_int(block_size * sizeof(value_type2));
  const unsigned int array_size3 = align_size_to_int(2 * block_size * sizeof(value_type5));

  return sizeof(int) * (array_size1 + array_size2 + array_size3);
} // end get_merge_kernel_per_block_dynamic_smem_usage()


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2, 
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename RandomAccessIterator5,
         typename StrictWeakOrdering,
         typename Size>
__global__ void merge_kernel(const RandomAccessIterator1 first1, 
                             const RandomAccessIterator1 last1,
                             const RandomAccessIterator2 first2,
                             const RandomAccessIterator2 last2,
                             RandomAccessIterator3 splitter_ranks1,
                             RandomAccessIterator4 splitter_ranks2,
                             const RandomAccessIterator5 result,
                             StrictWeakOrdering comp,
                             Size num_merged_partitions)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type1;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type value_type2;
  typedef typename thrust::iterator_value<RandomAccessIterator5>::type value_type5;

  // allocate shared storage
  const unsigned int array_size1 = align_size_to_int(blockDim.x * sizeof(value_type1));
  const unsigned int array_size2 = align_size_to_int(blockDim.x * sizeof(value_type2));
  const unsigned int array_size3 = align_size_to_int(2 * blockDim.x * sizeof(value_type5));
  int *_shared1 = extern_shared_ptr<int>();
  int *_shared2 = _shared1 + array_size1;
  int *_result  = _shared2 + array_size2;

  value_type1 *s_input1 = reinterpret_cast<value_type1*>(_shared1);
  value_type2 *s_input2 = reinterpret_cast<value_type2*>(_shared2);
  value_type5 *s_result = reinterpret_cast<value_type5*>(_result);

  // advance splitter iterators
  splitter_ranks1 += blockIdx.x;
  splitter_ranks2 += blockIdx.x;

  for(Size partition_idx = blockIdx.x;
      partition_idx < num_merged_partitions;
      partition_idx   += gridDim.x,
      splitter_ranks1 += gridDim.x,
      splitter_ranks2 += gridDim.x)
  {
    RandomAccessIterator1 input_begin1 = first1;
    RandomAccessIterator1 input_end1   = last1;
    RandomAccessIterator2 input_begin2 = first2;
    RandomAccessIterator2 input_end2   = last2;

    RandomAccessIterator5 output_begin = result;

    // find the end of the input if this is not the last block
    // the end of merged partition i is at splitter_ranks1[i] + splitter_ranks2[i]
    if(partition_idx != num_merged_partitions - 1)
    {
      RandomAccessIterator3 rank1 = splitter_ranks1;
      RandomAccessIterator4 rank2 = splitter_ranks2;

      input_end1 = first1 + dereference(rank1);
      input_end2 = first2 + dereference(rank2);
    }

    // find the beginning of the input and output if this is not the first partition
    // merged partition i begins at splitter_ranks1[i-1] + splitter_ranks2[i-1]
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
      output_begin += dereference(rank2);
    }

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

      block::merge(s_input1, s_input1 + s_input1_size,
                   s_input2, s_input2 + s_input2_size,
                   s_result,
                   comp);

      __syncthreads();

      // store to gmem
      output_begin = block::copy(s_result, s_result + s_input1_size + s_input2_size, output_begin);
    }

    // simply copy any remaining input
    block::copy(input_begin2, input_end2, block::copy(input_begin1, input_end1, output_begin));
  } // end for partition
} // end merge_kernel


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename Compare,
         typename Size1,
         typename Size2>
  void get_merge_splitter_ranks(RandomAccessIterator1 first1,
                                RandomAccessIterator1 last1,
                                RandomAccessIterator2 first2,
                                RandomAccessIterator2 last2,
                                RandomAccessIterator3 splitter_ranks1,
                                RandomAccessIterator4 splitter_ranks2,
                                Compare comp,
                                Size1 partition_size,
                                Size2 num_splitters_from_each_range)
{
  typedef typename thrust::iterator_difference<RandomAccessIterator1>::type difference1;
  typedef typename thrust::iterator_difference<RandomAccessIterator2>::type difference2;

  const difference1 num_elements1 = last1 - first1;
  const difference2 num_elements2 = last2 - first2;

  // zip up the ranges with a counter to disambiguate repeated elements during rank-finding
  typedef thrust::tuple<RandomAccessIterator1,thrust::counting_iterator<difference1> > iterator_tuple1;
  typedef thrust::tuple<RandomAccessIterator2,thrust::counting_iterator<difference2> > iterator_tuple2;
  typedef thrust::zip_iterator<iterator_tuple1> iterator_and_counter1;
  typedef thrust::zip_iterator<iterator_tuple2> iterator_and_counter2;

  iterator_and_counter1 first_and_counter1 =
    thrust::make_zip_iterator(thrust::make_tuple(first1, thrust::make_counting_iterator<difference1>(0)));
  iterator_and_counter1 last_and_counter1 = first_and_counter1 + num_elements1;

  // make the second range begin counting at num_elements1 so they sort after elements from the first range when ambiguous
  iterator_and_counter2 first_and_counter2 =
    thrust::make_zip_iterator(thrust::make_tuple(first2, thrust::make_counting_iterator<difference2>(num_elements1)));
  iterator_and_counter2 last_and_counter2 = first_and_counter2 + num_elements2;

  // take into account the tuples when comparing
  typedef compare_first_less_second<Compare> splitter_compare;

  using namespace thrust::detail::device::cuda::detail;
  return get_set_operation_splitter_ranks(first_and_counter1, last_and_counter1,
                                          first_and_counter2, last_and_counter2,
                                          splitter_ranks1,
                                          splitter_ranks2,
                                          splitter_compare(comp),
                                          partition_size,
                                          num_splitters_from_each_range);
} // end get_merge_splitter_ranks()


} // end namespace merge_detail


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2, 
	 typename RandomAccessIterator3,
         typename Compare>
RandomAccessIterator3 merge(RandomAccessIterator1 first1,
                            RandomAccessIterator1 last1,
                            RandomAccessIterator2 first2,
                            RandomAccessIterator2 last2,
                            RandomAccessIterator3 result,
                            Compare comp)
{
  typedef typename thrust::iterator_difference<RandomAccessIterator1>::type difference1;
  typedef typename thrust::iterator_difference<RandomAccessIterator2>::type difference2;

  const difference1 num_elements1 = last1 - first1;
  const difference2 num_elements2 = last2 - first2;

  // check for trivial problem
  if(num_elements1 == 0 && num_elements2 == 0)
    return result;
  else if(num_elements2 == 0)
    return thrust::copy(first1, last1, result);
  else if(num_elements1 == 0)
    return thrust::copy(first2, last2, result);

  using namespace merge_detail;
  using namespace thrust::detail;

  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type;
  
  // prefer large blocks to keep the partitions as large as possible
  const size_t block_size =
    arch::max_blocksize_subject_to_smem_usage(merge_kernel<
                                                RandomAccessIterator1,
                                                RandomAccessIterator2,
                                                typename raw_buffer<difference1,cuda_device_space_tag>::iterator,
                                                typename raw_buffer<difference2,cuda_device_space_tag>::iterator,
                                                RandomAccessIterator3,
                                                Compare,
                                                size_t
                                              >,
                                              get_merge_kernel_per_block_dynamic_smem_usage<
                                                RandomAccessIterator1,
                                                RandomAccessIterator2,
                                                RandomAccessIterator3
                                              >);

  const size_t partition_size = block_size;
  const difference1 num_partitions = ceil_div(num_elements1, partition_size);
  const difference1 num_splitters_from_each_range  = num_partitions - 1;
  size_t num_merged_partitions = 2 * num_splitters_from_each_range + 1;

  // allocate storage for splitter ranks
  raw_buffer<difference1, cuda_device_space_tag> splitter_ranks1(2 * num_splitters_from_each_range);
  raw_buffer<difference2, cuda_device_space_tag> splitter_ranks2(2 * num_splitters_from_each_range);

  // select some splitters and find the rank of each splitter in the other range
  // XXX it's possible to fuse rank-finding with the merge_kernel below
  //     this eliminates the temporary buffers splitter_ranks1 & splitter_ranks2
  //     but this spills to lmem and causes a 10x speeddown
  get_merge_splitter_ranks(first1,last1,
                           first2,last2,
                           splitter_ranks1.begin(),
                           splitter_ranks2.begin(),
                           comp,
                           partition_size,
                           num_splitters_from_each_range);

  // maximize the number of blocks we can launch
  size_t max_blocks = thrust::get<0>(thrust::detail::device::cuda::arch::max_grid_dimensions());
  size_t num_blocks = thrust::min(num_merged_partitions, max_blocks);

  merge_detail::merge_kernel<<<num_blocks, (unsigned int) block_size, get_merge_kernel_per_block_dynamic_smem_usage<RandomAccessIterator1,RandomAccessIterator2,RandomAccessIterator3>(block_size)>>>( 
  	first1, last1,
        first2, last2,
  	splitter_ranks1.begin(),
  	splitter_ranks2.begin(),
  	result, 
  	comp,
        num_merged_partitions);
  synchronize_if_enabled("merge_kernel");

  return result + num_elements1 + num_elements2;
} // end merge

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

