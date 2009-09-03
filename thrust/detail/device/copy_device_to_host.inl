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


/*! \file copy_device_to_device.h
 *  \brief Device implementations for copying on the device.
 */

#pragma once

#include <algorithm>          // for std::copy

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/raw_buffer.h>

#include <thrust/detail/device/trivial_copy.h>
#include <thrust/detail/type_traits.h>

namespace thrust
{

namespace detail
{

namespace device
{

// random access device to general host case
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy_device_to_host(InputIterator begin,
                                     InputIterator end,
                                     OutputIterator result,
                                     thrust::random_access_traversal_tag, 
                                     thrust::incrementable_traversal_tag)
{
    //std::cerr << std::endl;
    //std::cerr << "general copy_device_to_host(): InputIterator: " << typeid(InputIterator).name() << std::endl;
    //std::cerr << "general copy_device_to_host(): OutputIterator: " << typeid(OutputIterator).name() << std::endl;

    // allocate temporary storage
    typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;
    typename thrust::iterator_traits<InputIterator>::difference_type n = thrust::distance(begin, end);

    thrust::detail::raw_buffer<InputType,host_space_tag> temp(begin,end);
    result = thrust::copy(temp.begin(), temp.end(), result);

    return result;
}

// trivial device to host copy
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy_device_to_host(InputIterator begin,
                                     InputIterator end,
                                     OutputIterator result,
                                     thrust::random_access_traversal_tag,
                                     thrust::random_access_traversal_tag,
                                     true_type)
{
  //std::cerr << std::endl;
  //std::cerr << "random access copy_device_to_host(): trivial" << std::endl;
  //std::cerr << "general copy_device_to_host(): InputIterator: " << typeid(InputIterator).name() << std::endl;
  //std::cerr << "general copy_device_to_host(): OutputIterator: " << typeid(OutputIterator).name() << std::endl;
  
  // how many elements to copy?
  typename thrust::iterator_traits<OutputIterator>::difference_type n = end - begin;

  // what is the output type?
  typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

  trivial_copy_device_to_host(raw_pointer_cast(&*result), raw_pointer_cast(&*begin),  n * sizeof(OutputType));

  return result + n;
}

namespace detail
{

// random access non-trivial device iterator to random access host iterator
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator non_trivial_random_access_copy_device_to_host(InputIterator begin,
                                                               InputIterator end,
                                                               OutputIterator result,
                                                               false_type) // InputIterator is non-trivial
{
  // copy the input to a temporary device buffer of OutputType
  typedef typename thrust::iterator_value<OutputIterator>::type OutputType;

  typename thrust::iterator_difference<InputIterator>::type n = thrust::distance(begin,end);

  // allocate temporary storage
  thrust::detail::raw_buffer<OutputType,device_space_tag> temp(begin, end);
  result = thrust::copy(temp.begin(), temp.end(), result);

  return result;
}

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator non_trivial_random_access_copy_device_to_host(InputIterator begin,
                                                               InputIterator end,
                                                               OutputIterator result,
                                                               true_type) // InputIterator is trivial
{
  // copy the input to a temporary host buffer of InputType
  typedef typename thrust::iterator_value<InputIterator>::type InputType;

  typename thrust::iterator_difference<InputIterator>::type n = thrust::distance(begin,end);

  // allocate temporary storage
  thrust::detail::raw_buffer<InputType,host_space_tag> temp(n);

  // force a trivial copy
  thrust::detail::device::trivial_copy_device_to_host(raw_pointer_cast(&*temp.begin()), raw_pointer_cast(&*begin), n * sizeof(InputType));

  // finally, copy to the result
  result = thrust::copy(temp.begin(), temp.end(), result);

  return result;
}

} // end detail


// random access device iterator to random access host iterator with non-trivial copy
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy_device_to_host(InputIterator begin,
                                     InputIterator end,
                                     OutputIterator result,
                                     thrust::random_access_traversal_tag,
                                     thrust::random_access_traversal_tag,
                                     false_type)
{
  //std::cerr << std::endl;
  //std::cerr << "random access copy_device_to_host(): non-trivial" << std::endl;
  //std::cerr << "general copy_device_to_host(): InputIterator: " << typeid(InputIterator).name() << std::endl;
  //std::cerr << "general copy_device_to_host(): OutputIterator: " << typeid(OutputIterator).name() << std::endl;

  // dispatch a non-trivial random access device to host copy based on whether or not the InputIterator is trivial
  return detail::non_trivial_random_access_copy_device_to_host(begin, end, result,
      typename thrust::detail::is_trivial_iterator<InputIterator>::type());
}

// random access device iterator to random access host iterator
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy_device_to_host(InputIterator begin,
                                     InputIterator end,
                                     OutputIterator result,
                                     thrust::random_access_traversal_tag input_traversal,
                                     thrust::random_access_traversal_tag output_traversal)
{
  // dispatch on whether this is a trivial copy
  return copy_device_to_host(begin, end, result, input_traversal, output_traversal,
          typename is_trivial_copy<InputIterator,OutputIterator>::type());
}


/////////////////
// Entry Point //
/////////////////

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy_device_to_host(InputIterator begin, 
                                     InputIterator end, 
                                     OutputIterator result)
{
  return copy_device_to_host(begin, end, result, 
          typename thrust::iterator_traversal<InputIterator>::type(),
          typename thrust::iterator_traversal<OutputIterator>::type());
}

} // end namespace device

} // end namespace detail

} // end namespace thrust

