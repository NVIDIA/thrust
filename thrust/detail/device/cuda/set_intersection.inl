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

/*! \file set_intersection.inl
 *  \brief CUDA implementation of set_intersection,
 *         based on Gregory Diamos' original implementation.
 */

#include <thrust/detail/config.h>

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

// TODO use thrust/detail/device/ where possible
#include <thrust/iterator/iterator_traits.h>
#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/distance.h>
#include <thrust/extrema.h>

#include <thrust/detail/device/dereference.h>

#include <thrust/detail/device/cuda/block/copy.h>
#include <thrust/detail/device/cuda/block/inclusive_scan.h>
#include <thrust/detail/device/cuda/block/merging_sort.h>

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

template<unsigned int N>
  struct align_size_to_int
{
  static const unsigned int value = (N / sizeof(int)) + ((N % sizeof(int)) ? 1 : 0);
};

template<unsigned int block_size,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
__device__ unsigned int shared_memory_intersection(RandomAccessIterator results,
                                                   RandomAccessIterator smaller, 
                                                   unsigned int smallerSize,
                                                   RandomAccessIterator larger,
                                                   unsigned int largerSize,
                                                   StrictWeakOrdering comp)
{	
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;

  value_type element;
  RandomAccessIterator match = larger;
  unsigned int index = 0;
  unsigned int maxIndex = 0;
  bool keysMatch = false;
  unsigned int id = threadIdx.x;
  
  if(id < smallerSize)
  {
    element = smaller[id];

    // roll our own binary_search
    thrust::detail::device::cuda::block::lower_bound_workaround(larger, larger + largerSize, element, comp, match);
    keysMatch = (match != (larger + largerSize) && !comp(element, *match));
  }

  __shared__ unsigned int shared[block_size];
  shared[threadIdx.x] = keysMatch;
  __threadfence_block();

  thrust::detail::device::cuda::block::inplace_inclusive_scan<block_size>(shared, thrust::plus<int>());

  maxIndex = shared[block_size-1];

  if(threadIdx.x == 0)
  {
    index = 0;
  }
  else
  {
    index = shared[threadIdx.x - 1];
  }
  
  if(keysMatch)
  {		
    results[index] = element;
  }
  
  __syncthreads();
  
  return maxIndex;
}


template<unsigned int block_size,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2, 
	 typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename RandomAccessIterator5,
         typename RandomAccessIterator6,
         typename StrictWeakOrdering>
__global__ void set_intersection_kernel(RandomAccessIterator1 first1, 
                                        RandomAccessIterator1 last1,
                                        RandomAccessIterator2 first2,
                                        RandomAccessIterator2 last2,
                                        RandomAccessIterator3 result,
                                        RandomAccessIterator4 partition_begin_indices1, 
                                        RandomAccessIterator5 partition_begin_indices2, 
                                        RandomAccessIterator6 result_partition_sizes, 
                                        StrictWeakOrdering comp)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type;
  
  const unsigned int block_idx  = blockIdx.x;
  const unsigned int thread_idx = threadIdx.x;

  // advance iterators
  partition_begin_indices1 += block_idx;
  partition_begin_indices2 += block_idx;
  
  // find the ends of our partition if this is not the last block
  if(block_idx != gridDim.x - 1)
  {
    RandomAccessIterator4 temp1 = partition_begin_indices1 + 1;
    RandomAccessIterator5 temp2 = partition_begin_indices2 + 1;

    last1 = first1 + dereference(temp1);
    last2 = first2 + dereference(temp2);
  }
  
  // point to the beginning of our partition in each range
  first1 += dereference(partition_begin_indices1);
  first2 += dereference(partition_begin_indices2);
  result += dereference(partition_begin_indices1);

  const unsigned int array_size = align_size_to_int<block_size * sizeof(value_type)>::value;
  __shared__ int _first2[array_size];
  __shared__ int _first1[array_size];		
  __shared__ int _result[array_size];
  
  value_type* s_first2 = reinterpret_cast<value_type*>(_first2);
  value_type* s_first1 = reinterpret_cast<value_type*>(_first1);
  value_type* s_result = reinterpret_cast<value_type*>(_result);
  
  bool fetch2 = true;
  bool fetch1 = true;
  
  unsigned int size1 = 0;
  unsigned int size2 = 0;

  typename thrust::iterator_value<RandomAccessIterator6>::type result_partition_size(0);
  	
  while(true)
  {
    // XXX these copies are probably frequently the same length and could
    //     be merged
    if(fetch1)
    {
      RandomAccessIterator1 end = thrust::min(first1 + block_size, last1);
      size1 = end - first1;
      thrust::detail::device::cuda::block::copy(first1, end, s_first1);
      first1 = end;
    }

    if(fetch2)
    {
      RandomAccessIterator2 end = thrust::min(first2 + block_size, last2);
      size2 = end - first2;
      thrust::detail::device::cuda::block::copy(first2, end, s_first2);
      first2 = end;
    }
    
    __syncthreads();

    // XXX bring these into registers rather than do multiple shared loads?
    bool larger1 = comp(s_first2[size2 - 1], s_first1[size1 - 1]);
    bool larger2 = comp(s_first1[size1 - 1], s_first2[size2 - 1]);
    
    value_type* larger       = larger2 ? s_first2  : s_first1;
    unsigned int largerSize  = larger2 ? size2     : size1;
    value_type* smaller      = larger2 ? s_first1  : s_first2;
    unsigned int smallerSize = larger2 ? size1     : size2;
    
    fetch1 = !larger1;
    fetch2 = !larger2;
    
    // XXX this spot is probably the point of parameterization for other related algorithms
    unsigned int num_joined_elements = shared_memory_intersection<block_size>(s_result, smaller, smallerSize, larger, largerSize, comp);
    
    // copy out to result, advance iterator
    result = thrust::detail::device::cuda::block::copy(s_result, s_result + num_joined_elements, result);

    // increment size of the output
    result_partition_size += num_joined_elements;
    
    if(fetch1 && first1 >= last1) break;
    if(fetch2 && first2 >= last2) break;
  }
  
  if(thread_idx == 0)
  {
    dereference(result_partition_sizes,block_idx) = result_partition_size;
  }
}


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4>
__global__
void grouped_gather(RandomAccessIterator1 result,
                    RandomAccessIterator2 first,
                    RandomAccessIterator3 indices_begin,
                    RandomAccessIterator4 size_begin)
{
  using namespace thrust::detail::device;

  // advance input
  indices_begin += blockIdx.x;
  first += dereference(indices_begin);
  size_begin += blockIdx.x;
  
  typename thrust::iterator_value<RandomAccessIterator3>::type size = dereference(size_begin);
  
  if(blockIdx.x != 0)
  {
    RandomAccessIterator4 prev_size = size_begin - 1;

    // advance output
    result += dereference(prev_size);

    // compute size
    size -= dereference(prev_size);
  }
  
  thrust::detail::device::cuda::block::copy(first, first + size, result);
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
  typedef typename thrust::iterator_difference<RandomAccessIterator1>::type difference1;
  typedef typename thrust::iterator_difference<RandomAccessIterator2>::type difference2;

  const difference1 num_elements1 = last1 - first1;
  const difference2 num_elements2 = last2 - first2;

  // check for trivial problem
  if(num_elements1 == 0 || num_elements2 == 0)
    return result;

  using namespace set_intersection_detail;
  using namespace thrust::detail;

  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type;
  
  const size_t block_size = 128;
  const size_t partition_size = 1024;
  
  const difference1 num_partitions = ceil_div(num_elements1, partition_size);

  // create the range [0, partition_size, 2*partition_size, 3*partition_size, ...]
  typedef thrust::counting_iterator<difference1, cuda_device_space_tag> counter1;
  thrust::transform_iterator< mult_by<difference1>, counter1>
    partition_begin_indices1
      = thrust::make_transform_iterator(counter1(0), mult_by<difference1>(partition_size));
  
  // XXX we could encapsulate this gather in a permutation_iterator
  raw_buffer<value_type, cuda_device_space_tag> partition_values(num_partitions);
  thrust::next::gather(partition_begin_indices1, partition_begin_indices1 + partition_values.size(),
                       first1,
                       partition_values.begin());
  
  raw_buffer<difference2, cuda_device_space_tag> partition_begin_indices2(num_partitions);
  
  thrust::lower_bound(first2, last2,
                      partition_values.begin(), partition_values.end(), 
                      partition_begin_indices2.begin(), comp);
  
  raw_buffer<difference1, cuda_device_space_tag> result_partition_sizes(num_partitions);
  raw_buffer< value_type, cuda_device_space_tag> temp_result(num_elements1);
  
  set_intersection_detail::set_intersection_kernel<block_size><<< num_partitions, block_size >>>( 
  	first1, last1,
        first2, last2,
        temp_result.begin(), 
  	partition_begin_indices1,
  	partition_begin_indices2.begin(),
  	result_partition_sizes.begin(), 
  	comp);
  
  thrust::inclusive_scan(result_partition_sizes.begin(), result_partition_sizes.end(), result_partition_sizes.begin());
  
  set_intersection_detail::grouped_gather<<< num_partitions, block_size >>>( 
  	result,
  	temp_result.begin(),
  	partition_begin_indices1,
  	result_partition_sizes.begin());
  
  return result + result_partition_sizes[num_partitions - 1];
} // end set_intersection

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

#endif // THRUST_DEVICE_COMPILER_NVCC

