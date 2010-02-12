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
#include <thrust/unique.h>

#include <thrust/detail/device/dereference.h>

#include <thrust/detail/device/cuda/block/copy.h>
#include <thrust/detail/device/cuda/block/inclusive_scan.h>
#include <thrust/detail/device/cuda/scalar/rotate.h>
#include <thrust/detail/device/cuda/scalar/binary_search.h>
#include <thrust/pair.h>
#include <thrust/extrema.h>

//#define THRUST_DEBUG_SET_INTERSECTION

#ifdef THRUST_DEBUG_SET_INTERSECTION
// XXX remove me
#include <cassert>
#include <thrust/reduce.h>
#include <thrust/cuPrintf.cu>
#endif

namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{

namespace block
{

template<unsigned int block_size,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename StrictWeakOrdering>
__device__
thrust::tuple<
  RandomAccessIterator1,
  RandomAccessIterator2,
  RandomAccessIterator3
>

  set_intersection(RandomAccessIterator1 first1,
                   RandomAccessIterator1 last1,
                   RandomAccessIterator2 first2,
                   RandomAccessIterator2 last2,
                   RandomAccessIterator3 result,
                   StrictWeakOrdering comp)
{
  bool do_output_element = false;

  // these variables are needed outside of the branch for the last part of the algorithm
  // so declare them here
  thrust::pair<RandomAccessIterator2,RandomAccessIterator2> matches2 = thrust::make_pair(first2,first2);

  typename thrust::iterator_difference<RandomAccessIterator1>::type size_of_partial_intersection(0);

  // for each element in the first range, search the second range, looking for a match
  if(first1 + threadIdx.x < last1)
  {
    // not only must we figure out if our element exists in the second range, we need to
    // 1. figure out how many m copies of our element exists in the first range
    // 2. figure out how many n copies of our element exists in the second range
    // 3. keep the element if its rank < min(m,n)

    thrust::pair<RandomAccessIterator1,RandomAccessIterator1> matches1 = scalar::equal_range(first1, last1, first1[threadIdx.x], comp);
    matches2 = scalar::equal_range(first2, last2, first1[threadIdx.x], comp);

    size_of_partial_intersection = thrust::min(matches1.second - matches1.first,
                                               matches2.second - matches2.first);

    // compute the rank of the element in the first range
    unsigned int element_rank = threadIdx.x - (matches1.first - first1);

    do_output_element = (element_rank < size_of_partial_intersection);
  }

  // copy matches to the result

  // record, for each thread, whether we should output its element
  __shared__ unsigned int shared[block_size];
  shared[threadIdx.x] = do_output_element;

  // a shared copy_if operation follows:

  block::inplace_inclusive_scan<block_size>(shared, thrust::plus<unsigned int>());

  unsigned int size_of_total_intersection = shared[block_size-1];

  // find the index to write our element
  unsigned int output_index = 0;
  if(threadIdx.x > 0)
  {
    output_index = shared[threadIdx.x-1];
  }

  if(do_output_element)
  {
    result[output_index] = first1[threadIdx.x];
  }

  // find the last element of both ranges to be consumed, and point to the next element
  __shared__ unsigned int one_past_last_element_consumed[2];
  if(threadIdx.x == 0)
  {
    one_past_last_element_consumed[0] = 0;
    one_past_last_element_consumed[1] = 0;
  }
  __syncthreads();

  if(do_output_element && shared[threadIdx.x] == size_of_total_intersection)
  {
    one_past_last_element_consumed[0] = threadIdx.x+1;
    one_past_last_element_consumed[1] = (matches2.first + size_of_partial_intersection) - first2;
  }
  __syncthreads();

  return thrust::make_tuple(first1 + one_past_last_element_consumed[0],
                            first2 + one_past_last_element_consumed[1],
                            result + size_of_total_intersection);
} // end set_intersection

} // end block

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
                                        StrictWeakOrdering comp,
                                        unsigned int *debug)
{
#ifdef THRUST_DEBUG_SET_INTERSECTION
  cuPrintfRestrict(0,CUPRINTF_UNRESTRICTED);
#endif

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
  
  value_type* s_result = reinterpret_cast<value_type*>(_result);

  // keep two ranges in shared memory
  pair<value_type*,value_type*> s_range1 = make_pair(reinterpret_cast<value_type*>(_first1),
                                                     reinterpret_cast<value_type*>(_first1));

  pair<value_type*,value_type*> s_range2 = make_pair(reinterpret_cast<value_type*>(_first2),
                                                     reinterpret_cast<value_type*>(_first2));
  
  bool fetch2 = true;
  bool fetch1 = true;
  
  typename thrust::iterator_value<RandomAccessIterator6>::type result_partition_size(0);
  	
  while(true)
  {
    // XXX these copies are probably frequently the same length and could
    //     be merged

    // bring the segment from the first range into shared memory
    if(fetch1)
    {
      // push remaining input from the previous iteration to the front of the storage
      // XXX rotate is actually redundant -- we don't need to keep [first,middle)
      //__syncthreads()
      //if(threadIdx.x == 0)
      //{
      //  scalar::rotate(reinterpret_cast<value_type*>(_first1), s_range1.first, s_range1.second);
      //}
      //__syncthreads();

      RandomAccessIterator1 end = thrust::min(first1 + block_size, last1);

#ifdef THRUST_DEBUG_SET_INTERSECTION
      cuPrintf("Fetching %d from left side\n", (end - first1));
#endif

      // reset s_range1 to point to the beginning of its shared storage
      s_range1.first = reinterpret_cast<value_type*>(_first1);

      // copy from the input into shared mem
      s_range1.second = thrust::detail::device::cuda::block::copy(first1, end, s_range1.first);

      // bump the input iterator
      first1 = end;
    }

    // bring the segment from the second range into shared memory
    if(fetch2)
    {
      RandomAccessIterator2 end = thrust::min(first2 + block_size, last2);

#ifdef THRUST_DEBUG_SET_INTERSECTION
      cuPrintf("Fetching %d from right side\n", (end - first2));
#endif

      // reset s_range2 to point to the beginning of its shared storage
      s_range2.first = reinterpret_cast<value_type*>(_first2);

      // copy from the input into shared mem
      s_range2.second = thrust::detail::device::cuda::block::copy(first2, end, s_range2.first);

      // bump the input iterator
      first2 = end;
    }
    
    __syncthreads();

    // XXX bring these into registers rather than do multiple shared loads?
    bool greater1 = comp(s_range2.second[-1], s_range1.second[-1]);
    bool greater2 = comp(s_range1.second[-1], s_range2.second[-1]);
    
    pair<value_type*,value_type*> &s_range_with_greater_bound = greater1 ? s_range1 : s_range2;
    pair<value_type*,value_type*> &s_range_with_lesser_bound  = greater1 ? s_range2 : s_range1;
    
    fetch1 = !greater1;
    fetch2 = !greater2;
    
#ifdef THRUST_DEBUG_SET_INTERSECTION
    if((s_range_with_lesser_bound.second - s_range_with_lesser_bound.first) > block_size)
    {
      cuPrintf("ERROR: range_with_lesser_bound.size(): %d\n", (s_range_with_lesser_bound.second - s_range_with_lesser_bound.first));
    }

    if((s_range_with_greater_bound.second - s_range_with_greater_bound.first) > block_size)
    {
      cuPrintf("ERROR: range_with_greater_bound.size(): %d\n", (s_range_with_greater_bound.second - s_range_with_greater_bound.first));
    }

    cuPrintf("lesser  range: [%d, %d]\n", s_range_with_lesser_bound.first[0],  s_range_with_lesser_bound.second[-1]);
    cuPrintf("greater range: [%d, %d]\n", s_range_with_greater_bound.first[0], s_range_with_greater_bound.second[-1]);
#endif

    // XXX this spot is probably the point of parameterization for other related algorithms
    value_type *s_result_end;

    // this operation simultaneously updates shared ranges' begin pointers past the last common input element
    tie(s_range_with_lesser_bound.first, s_range_with_greater_bound.first, s_result_end)
      = block::set_intersection<block_size>(s_range_with_lesser_bound.first,  s_range_with_lesser_bound.second,
                                            s_range_with_greater_bound.first, s_range_with_greater_bound.second,
                                            s_result, comp);
    
    // copy out to result, advance iterator
    result = thrust::detail::device::cuda::block::copy(s_result, s_result_end, result);

#ifdef THRUST_DEBUG_SET_INTERSECTION
    if(s_result_end > s_result)
      cuPrintf("Output %d, last result output: %d\n", (s_result_end - s_result), dereference(result-1));
    else
      cuPrintf("Nothing output that round\n");
#endif

    // increment size of the output
    result_partition_size += (s_result_end - s_result);
    
    if(fetch1 && first1 >= last1) break;
    if(fetch2 && first2 >= last2) break;
  }
  
  if(thread_idx == 0)
  {
    result_partition_sizes += block_idx;
    dereference(result_partition_sizes) = result_partition_size;
  }
}


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4>
__global__
void grouped_gather(RandomAccessIterator1 result,
                    RandomAccessIterator2 first,
                    RandomAccessIterator3 input_segment_begin_indices,
                    RandomAccessIterator4 output_segment_end_indices)
{
  using namespace thrust::detail::device;

  // advance iterators
  input_segment_begin_indices += blockIdx.x;
  output_segment_end_indices  += blockIdx.x;

  // point at the beginning of this block's input
  first += dereference(input_segment_begin_indices);
  
  // initialize the size of the segment to read
  typename thrust::iterator_value<RandomAccessIterator4>::type size = dereference(output_segment_end_indices);
  
  if(blockIdx.x != 0)
  {
    RandomAccessIterator4 ptr_to_previous_segment_end_index = output_segment_end_indices;
    --ptr_to_previous_segment_end_index;

    // get the end of the previous segment
    typename thrust::iterator_value<RandomAccessIterator4>::type previous_segment_end_index = dereference(ptr_to_previous_segment_end_index);

    // point result at our segment's begin index
    result += previous_segment_end_index;

    // compute our segment's size: the difference between our end index and the previous segment's end index
    size -= previous_segment_end_index;
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
  
  difference1 num_partitions = ceil_div(num_elements1, partition_size);

#ifdef THRUST_DEBUG_SET_INTERSECTION
  std::cerr << "num_partitions: " << num_partitions << std::endl;
#endif

  // create the range [0, partition_size, 2*partition_size, 3*partition_size, ...]
  typedef thrust::counting_iterator<difference1, cuda_device_space_tag> counter1;
  thrust::transform_iterator< mult_by<difference1>, counter1>
    partition_begin_indices1_guess
      = thrust::make_transform_iterator(counter1(0), mult_by<difference1>(partition_size));
  // XXX we could encapsulate this gather in a permutation_iterator
  raw_buffer<value_type, cuda_device_space_tag> partition_values(num_partitions);
  thrust::next::gather(partition_begin_indices1_guess, partition_begin_indices1_guess + partition_values.size(),
                       first1,
                       partition_values.begin());

  typename raw_buffer<value_type, cuda_device_space_tag>::iterator end = 
    thrust::unique(partition_values.begin(), partition_values.end());
  num_partitions = end - partition_values.begin();
  
  raw_buffer<difference1, cuda_device_space_tag> partition_begin_indices1(num_partitions);
  raw_buffer<difference2, cuda_device_space_tag> partition_begin_indices2(num_partitions);

  thrust::lower_bound(first1, last1,
                      partition_values.begin(), partition_values.end(), 
                      partition_begin_indices1.begin(), comp);
  
  thrust::lower_bound(first2, last2,
                      partition_values.begin(), partition_values.end(), 
                      partition_begin_indices2.begin(), comp);

  raw_buffer<difference1, cuda_device_space_tag> result_partition_sizes(num_partitions);
  raw_buffer< value_type, cuda_device_space_tag> temp_result(num_elements1);

  thrust::device_vector<unsigned int> debug(1, 0);

#ifdef THRUST_DEBUG_SET_INTERSECTION
  cudaPrintfInit();
#endif
  
  set_intersection_detail::set_intersection_kernel<block_size><<< num_partitions, block_size >>>( 
  	first1, last1,
        first2, last2,
        temp_result.begin(), 
  	partition_begin_indices1.begin(),
  	partition_begin_indices2.begin(),
  	result_partition_sizes.begin(), 
  	comp,
        thrust::raw_pointer_cast(&debug[0]));

#ifdef THRUST_DEBUG_SET_INTERSECTION
  cudaPrintfDisplay(stdout, true);
  cudaPrintfEnd();
#endif


  thrust::inclusive_scan(result_partition_sizes.begin(), result_partition_sizes.end(), result_partition_sizes.begin());

  // after the inclusive scan, we have the end of each segment
  raw_buffer<difference1, cuda_device_space_tag> &output_segment_end_indices = result_partition_sizes;
  
  set_intersection_detail::grouped_gather<<< num_partitions, block_size >>>( 
  	result,
  	temp_result.begin(),
  	partition_begin_indices1.begin(),
  	output_segment_end_indices.begin());

  return result + result_partition_sizes[num_partitions - 1];
} // end set_intersection

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

