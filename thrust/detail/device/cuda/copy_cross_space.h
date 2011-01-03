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

#pragma once

#include <thrust/copy.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/raw_buffer.h>
#include <thrust/detail/dispatch/is_trivial_copy.h>
#include <thrust/detail/device/cuda/trivial_copy.h>

namespace thrust
{

namespace detail
{

// XXX WAR circular #inclusion problem
template<typename,typename> class raw_buffer;

namespace device
{

namespace cuda
{


// general input to random access case
template<typename InputIterator,
         typename RandomAccessIterator>
  RandomAccessIterator copy_cross_space(InputIterator begin,
                                        InputIterator end,
                                        RandomAccessIterator result,
                                        thrust::incrementable_traversal_tag, 
                                        thrust::random_access_traversal_tag)
{
  //std::cerr << std::endl;
  //std::cerr << "general copy_host_to_device(): InputIterator: " << typeid(InputIterator).name() << std::endl;
  //std::cerr << "general copy_host_to_device(): OutputIterator: " << typeid(OutputIterator).name() << std::endl;

  typedef typename thrust::iterator_value<InputIterator>::type InputType;
  typedef typename thrust::iterator_space<InputIterator>::type InputSpace;

  // allocate temporary storage
  thrust::detail::raw_buffer<InputType, InputSpace> temp(begin,end);
  result = thrust::copy(temp.begin(), temp.end(), result);

  return result;
}

template<typename InputIterator,
         typename Size,
         typename RandomAccessIterator>
  RandomAccessIterator copy_cross_space_n(InputIterator first,
                                          Size n,
                                          RandomAccessIterator result,
                                          thrust::incrementable_traversal_tag, 
                                          thrust::random_access_traversal_tag)
{
  typedef typename thrust::iterator_value<InputIterator>::type InputType;
  typedef typename thrust::iterator_space<InputIterator>::type InputSpace;

  // allocate and copy to temporary storage in the input's space
  thrust::detail::raw_buffer<InputType, InputSpace> temp(n);
  thrust::copy_n(first, n, temp.begin());

  return thrust::copy(temp.begin(), temp.end(), result);
}


// random access to general output case
template<typename RandomAccessIterator,
         typename OutputIterator>
  OutputIterator copy_cross_space(RandomAccessIterator begin,
                                  RandomAccessIterator end,
                                  OutputIterator result,
                                  thrust::random_access_traversal_tag, 
                                  thrust::incrementable_traversal_tag)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type InputType;
  typedef typename thrust::iterator_space<OutputIterator>::type OutputSpace;

  // allocate temporary storage
  thrust::detail::raw_buffer<InputType,OutputSpace> temp(begin,end);
  result = thrust::copy(temp.begin(), temp.end(), result);

  return result;
}

template<typename RandomAccessIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_cross_space_n(RandomAccessIterator first,
                                    Size n,
                                    OutputIterator result,
                                    thrust::random_access_traversal_tag, 
                                    thrust::incrementable_traversal_tag)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type InputType;
  typedef typename thrust::iterator_space<OutputIterator>::type OutputSpace;

  // allocate and copy to temporary storage in the output's space
  thrust::detail::raw_buffer<InputType,OutputSpace> temp(n);
  thrust::copy_n(first, n, temp.begin());

  return thrust::copy(temp.begin(), temp.end(), result);
}


// trivial copy
template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  RandomAccessIterator2 copy_cross_space(RandomAccessIterator1 begin,
                                         RandomAccessIterator1 end,
                                         RandomAccessIterator2 result,
                                         thrust::random_access_traversal_tag,
                                         thrust::random_access_traversal_tag,
                                         true_type) // trivial copy
{
  //std::cerr << std::endl;
  //std::cerr << "random access copy_device_to_host(): trivial" << std::endl;
  //std::cerr << "general copy_device_to_host(): InputIterator: " << typeid(InputIterator).name() << std::endl;
  //std::cerr << "general copy_device_to_host(): OutputIterator: " << typeid(OutputIterator).name() << std::endl;
  
  // how many elements to copy?
  typename thrust::iterator_traits<RandomAccessIterator1>::difference_type n = end - begin;

  thrust::detail::device::cuda::trivial_copy_n(begin, n, result);

  return result + n;
}


namespace detail
{

// random access non-trivial iterator to random access iterator
template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  RandomAccessIterator2 non_trivial_random_access_copy_cross_space(RandomAccessIterator1 begin,
                                                                   RandomAccessIterator1 end,
                                                                   RandomAccessIterator2 result,
                                                                   false_type) // InputIterator is non-trivial
{
  // copy the input to a temporary input space buffer of OutputType
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type OutputType;
  typedef typename thrust::iterator_space<RandomAccessIterator1>::type InputSpace;

  // allocate temporary storage
  thrust::detail::raw_buffer<OutputType,InputSpace> temp(begin, end);
  result = thrust::copy(temp.begin(), temp.end(), result);

  return result;
}

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  RandomAccessIterator2 non_trivial_random_access_copy_cross_space(RandomAccessIterator1 begin,
                                                                   RandomAccessIterator1 end,
                                                                   RandomAccessIterator2 result,
                                                                   true_type) // InputIterator is trivial
{
  // copy the input to a temporary result space buffer of InputType
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type InputType;
  typedef typename thrust::iterator_space<RandomAccessIterator2>::type OutputSpace;

  typename thrust::iterator_difference<RandomAccessIterator1>::type n = thrust::distance(begin,end);

  // allocate temporary storage
  thrust::detail::raw_buffer<InputType,OutputSpace> temp(n);

  // force a trivial copy
  thrust::detail::device::cuda::trivial_copy_n(begin, n, temp.begin());

  // finally, copy to the result
  result = thrust::copy(temp.begin(), temp.end(), result);

  return result;
}

} // end detail


// random access iterator to random access host iterator with non-trivial copy
template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  RandomAccessIterator2 copy_cross_space(RandomAccessIterator1 begin,
                                         RandomAccessIterator1 end,
                                         RandomAccessIterator2 result,
                                         thrust::random_access_traversal_tag,
                                         thrust::random_access_traversal_tag,
                                         false_type)
{
  // dispatch a non-trivial random access cross space copy based on whether or not the InputIterator is trivial
  return detail::non_trivial_random_access_copy_cross_space(begin, end, result,
      typename thrust::detail::is_trivial_iterator<RandomAccessIterator1>::type());
}

// random access iterator to random access iterator
template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  RandomAccessIterator2 copy_cross_space(RandomAccessIterator1 begin,
                                         RandomAccessIterator1 end,
                                         RandomAccessIterator2 result,
                                         thrust::random_access_traversal_tag input_traversal,
                                         thrust::random_access_traversal_tag output_traversal)
{
  // dispatch on whether this is a trivial copy
  return copy_cross_space(begin, end, result, input_traversal, output_traversal,
          typename thrust::detail::dispatch::is_trivial_copy<RandomAccessIterator1,RandomAccessIterator2>::type());
}

template<typename RandomAccessIterator1,
         typename Size,
         typename RandomAccessIterator2>
  RandomAccessIterator2 copy_cross_space_n(RandomAccessIterator1 first,
                                           Size n,
                                           RandomAccessIterator2 result,
                                           thrust::random_access_traversal_tag input_traversal,
                                           thrust::random_access_traversal_tag output_traversal)
{
  // implement with copy_cross_space
  return copy_cross_space(first, first + n, result, input_traversal, output_traversal);
}

/////////////////
// Entry Point //
/////////////////

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy_cross_space(InputIterator begin, 
                                  InputIterator end, 
                                  OutputIterator result)
{
  return copy_cross_space(begin, end, result, 
          typename thrust::iterator_traversal<InputIterator>::type(),
          typename thrust::iterator_traversal<OutputIterator>::type());
}

template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_cross_space_n(InputIterator begin, 
                                    Size n, 
                                    OutputIterator result)
{
  return copy_cross_space_n(begin, n, result, 
          typename thrust::iterator_traversal<InputIterator>::type(),
          typename thrust::iterator_traversal<OutputIterator>::type());
}

} // end cuda

} // end device

} // end detail

} // end thrust

