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

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/detail/device/dereference.h>
#include <thrust/pair.h>
#include <thrust/extrema.h>
#include <thrust/detail/device/cuda/block/copy.h>
#include <thrust/detail/device/cuda/scalar/binary_search.h>
#include <thrust/detail/device/cuda/synchronize.h>

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
  struct align_size_to_int
{
  static const unsigned int value = (N / sizeof(int)) + ((N % sizeof(int)) ? 1 : 0);
};

template<unsigned int block_size,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2, 
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename RandomAccessIterator5,
         typename StrictWeakOrdering>
__launch_bounds__(block_size, 1)         
__global__ void merge_kernel(RandomAccessIterator1 first1, 
                             RandomAccessIterator1 last1,
                             RandomAccessIterator2 first2,
                             RandomAccessIterator2 last2,
                             RandomAccessIterator3 splitter_ranks1,
                             RandomAccessIterator4 splitter_ranks2,
                             RandomAccessIterator5 result,
                             StrictWeakOrdering comp)
{
  const unsigned int partition_idx  = blockIdx.x;

  // advance iterators
  splitter_ranks1 += partition_idx;
  splitter_ranks2 += partition_idx;

  // find the end of the input if this is not the last block
  // the end of merged partition i is at splitter_ranks1[i+1] + splitter_ranks2[i+1]
  if(partition_idx != gridDim.x - 1)
  {
    RandomAccessIterator3 temp1 = splitter_ranks1 + 1;
    RandomAccessIterator4 temp2 = splitter_ranks2 + 1;

    last1 = first1 + dereference(temp1);
    last2 = first2 + dereference(temp2);
  }

  // find the beginning of the input and output if this is not the first block
  // merged partition i begins at splitter_ranks1[i] + splitter_ranks2[i]
  if(partition_idx != 0)
  {
    RandomAccessIterator3 temp1 = splitter_ranks1;
    RandomAccessIterator4 temp2 = splitter_ranks2;

    // advance the input to point to the beginning
    first1 += dereference(temp1);
    first2 += dereference(temp2);

    // advance the result to point to the beginning of the output
    result += dereference(temp1);
    result += dereference(temp2);
  }

  typedef typename thrust::iterator_value<RandomAccessIterator5>::type value_type;

  // allocate shared storage
  const unsigned int array_size = align_size_to_int<block_size * sizeof(value_type)>::value;
  __shared__ int _shared1[array_size];
  __shared__ int _shared2[array_size];
  __shared__ int _result[align_size_to_int<2 * block_size * sizeof(value_type)>::value];

  value_type *s_input1 = reinterpret_cast<value_type*>(_shared1);
  value_type *s_input2 = reinterpret_cast<value_type*>(_shared2);
  value_type *s_result = reinterpret_cast<value_type*>(_result);

  typedef typename thrust::iterator_difference<RandomAccessIterator1>::type difference1;
  difference1 s_input1_size = 0;

  typedef typename thrust::iterator_difference<RandomAccessIterator2>::type difference2;
  difference2 s_input2_size = 0;

  // consume input into our storage over multiple rounds
  while(first1 < last1 && first2 < last2)
  {
    // fetch into the first segment if there's input left
    if(first1 < last1)
    {
      s_input1_size = thrust::min<difference1>(block_size, last1 - first1);

      block::copy(first1, first1 + s_input1_size, s_input1);
      first1 += s_input1_size;
    }

    // fetch into the second segment if there's input left
    if(first2 < last2)
    {
      s_input2_size = thrust::min<difference2>(block_size, last2 - first2);

      block::copy(first2, first2 + s_input2_size, s_input2);
      first2 += s_input2_size;
    }

    __syncthreads();

    // find the rank of each element in the other array
    unsigned int rank2 = 0;
    if(threadIdx.x < s_input1_size)
    {
      value_type x = s_input1[threadIdx.x];

      // lower_bound ensures that x sorts before any equivalent element of s_input2
      // this ensures stability
      rank2 = scalar::lower_bound(s_input2, s_input2 + s_input2_size, x, comp) - s_input2;
    }

    unsigned int rank1 = 0;
    if(threadIdx.x < s_input2_size)
    {
      value_type x = s_input2[threadIdx.x];

      // upper_bound ensures that x sorts after any equivalent element of s_input1
      // this ensures stability
      rank1 = scalar::upper_bound(s_input1, s_input1 + s_input1_size, x, comp) - s_input1;
    }

    __syncthreads();

    if(threadIdx.x < s_input1_size)
    {
      // scatter each element from input1
      s_result[threadIdx.x + rank2] = s_input1[threadIdx.x];
    }

    if(threadIdx.x < s_input2_size)
    {
      // scatter each element from input2
      s_result[threadIdx.x + rank1] = s_input2[threadIdx.x];
    }

    __syncthreads();

    // store to gmem
    result = block::copy(s_result, s_result + s_input1_size + s_input2_size, result);

    __syncthreads();
  }

  // simply copy any remaining input
  if(first1 < last1)
    block::copy(first1, last1, result);
  else if(first2 < last2)
    block::copy(first2, last2, result);
}


template<typename T>
struct mult_by
  : thrust::unary_function<T,T>
{
  T _value;
  
  mult_by(const T& v):_value(v){}
  
  __host__ __device__
  T operator()(const T& v) const
  {
    return _value * v;
  }
};

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
  
  const size_t block_size = 128;
  const size_t partition_size = block_size;
  
  const difference1 num_partitions = ceil_div(num_elements1, partition_size);

  // create the range [0, partition_size, 2*partition_size, 3*partition_size, ...]
  typedef thrust::counting_iterator<difference1, cuda_device_space_tag> counter;
  typedef thrust::transform_iterator<mult_by<difference1>, counter>    leapfrog_iterator;
  leapfrog_iterator splitter_ranks1
    = thrust::make_transform_iterator(counter(0), mult_by<difference1>(partition_size));

  // create the range [first1[0], first1[partition_size], first1[2*partition_size], ...]
  typedef thrust::permutation_iterator<RandomAccessIterator1, leapfrog_iterator> splitter_iterator;
  splitter_iterator splitters_begin
    = thrust::make_permutation_iterator(first1, splitter_ranks1);

  splitter_iterator splitters_end = splitters_begin + num_partitions;

  raw_buffer<difference2, cuda_device_space_tag> splitter_ranks2(num_partitions);

  // find the rank of each splitter in the second range
  thrust::lower_bound(first2, last2,
                      splitters_begin, splitters_end, 
                      splitter_ranks2.begin(), comp);

//  std::cerr << "splitter: " << std::endl;
//  for(int i = 0; i < num_partitions; ++i)
//  {
//    std::cerr << "splitters[" << i << "]: " << (char)splitters_begin[i] << std::endl;
//  }
//
//  std::cerr << "splitter_ranks1: " << std::endl;
//  for(int i = 0; i < num_partitions; ++i)
//  {
//    std::cerr << "splitter_ranks1[" << i << "]: " << splitter_ranks1[i] << std::endl;
//  }
//
//  std::cerr << "splitter_ranks2: " << std::endl;
//  for(int i = 0; i < num_partitions; ++i)
//  {
//    std::cerr << "splitter_ranks2[" << i << "]: " << splitter_ranks2[i] << std::endl;
//  }
//
//  std::cerr << "num_partitions: " << num_partitions << std::endl;
//  std::cerr << "block_size: " << block_size << std::endl;

  merge_detail::merge_kernel<block_size><<<(unsigned int) num_partitions, (unsigned int) block_size >>>( 
  	first1, last1,
        first2, last2,
  	splitter_ranks1,
  	splitter_ranks2.begin(),
  	result, 
  	comp);
  synchronize_if_enabled("merge_kernel");

  return result + num_elements1 + num_elements2;
} // end merge

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

