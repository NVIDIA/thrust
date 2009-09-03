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

#include <thrust/iterator/iterator_traits.h>

#include <thrust/distance.h>
#include <thrust/device_ptr.h>

#include <thrust/detail/device/trivial_copy.h>
#include <thrust/detail/raw_buffer.h>

namespace thrust
{

namespace detail
{

namespace device
{

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy_host_to_device(InputIterator begin,
                                     InputIterator end,
                                     OutputIterator result,
                                     thrust::incrementable_traversal_tag, 
                                     thrust::random_access_traversal_tag)
{
    //std::cerr << std::endl;
    //std::cerr << "general copy_host_to_device(): InputIterator: " << typeid(InputIterator).name() << std::endl;
    //std::cerr << "general copy_host_to_device(): OutputIterator: " << typeid(OutputIterator).name() << std::endl;

    // host container to device container
    typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;

    // allocate temporary storage
    thrust::detail::raw_buffer<InputType, host_space_tag> temp(begin,end);

    result = thrust::copy(temp.begin(), temp.end(), result);

    return result;
}

// host pointer to device pointer with trivial copy
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy_host_to_device(InputIterator begin,
                                     InputIterator end,
                                     OutputIterator result,
                                     thrust::random_access_traversal_tag,
                                     thrust::random_access_traversal_tag,
                                     true_type)
{
  //std::cerr << std::endl;
  //std::cerr << "random access copy_host_to_device(): trivial" << std::endl;
  //std::cerr << "general copy_host_to_device(): InputIterator: " << typeid(InputIterator).name() << std::endl;
  //std::cerr << "general copy_host_to_device(): OutputIterator: " << typeid(OutputIterator).name() << std::endl;
  
  // how many elements to copy?
  typename thrust::iterator_traits<OutputIterator>::difference_type n = end - begin;

  // what is the output type?
  typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

  trivial_copy_host_to_device(raw_pointer_cast(&*result), raw_pointer_cast(&*begin),  n * sizeof(OutputType));

  return result + n;
}


namespace detail
{

// random access non-trivial host iterator to random access device iterator
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator non_trivial_random_access_copy_host_to_device(InputIterator begin,
                                                               InputIterator end,
                                                               OutputIterator result,
                                                               false_type) // InputIterator is non-trivial
{
  // copy the input to a temporary host buffer of OutputType
  typedef typename thrust::iterator_value<OutputIterator>::type OutputType;

  typename thrust::iterator_difference<InputIterator>::type n = thrust::distance(begin,end);

  // allocate temporary storage
  thrust::detail::raw_buffer<OutputType, host_space_tag> temp(begin, end);

  result = thrust::copy(temp.begin(), temp.end(), result);

  return result;
}

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator non_trivial_random_access_copy_host_to_device(InputIterator begin,
                                                               InputIterator end,
                                                               OutputIterator result,
                                                               true_type) // InputIterator is trivial
{
  // copy the input to a temporary device buffer of InputType
  typedef typename thrust::iterator_value<InputIterator>::type InputType;

  typename thrust::iterator_difference<InputIterator>::type n = thrust::distance(begin,end);

  // allocate temporary storage
  thrust::detail::raw_buffer<InputType, device_space_tag> temp(n);

  // force a trivial copy
  thrust::detail::device::trivial_copy_host_to_device(raw_pointer_cast(&*temp.begin()), raw_pointer_cast(&*begin), n * sizeof(InputType));

  result = thrust::copy(temp.begin(), temp.end(), result);

  return result;
}

} // end detail


// random access host iterator to random access device iterator with non-trivial copy
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy_host_to_device(InputIterator begin,
                                     InputIterator end,
                                     OutputIterator result,
                                     thrust::random_access_traversal_tag,
                                     thrust::random_access_traversal_tag,
                                     false_type)
{
  //std::cerr << std::endl;
  //std::cerr << "random access copy_host_to_device(): non-trivial" << std::endl;
  //std::cerr << "general copy_host_to_device(): InputIterator: " << typeid(InputIterator).name() << std::endl;
  //std::cerr << "general copy_host_to_device(): OutputIterator: " << typeid(OutputIterator).name() << std::endl;

  // dispatch a non-trivial random access host to device copy based on whether or not the InputIterator is trivial
  return detail::non_trivial_random_access_copy_host_to_device(begin, end, result,
      typename thrust::detail::is_trivial_iterator<InputIterator>::type());
}

// random access host iterator to random access device iterator
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy_host_to_device(InputIterator begin,
                                     InputIterator end,
                                     OutputIterator result,
                                     thrust::random_access_traversal_tag input_traversal,
                                     thrust::random_access_traversal_tag output_traversal)
{
    // dispatch on whether this is a trivial copy
    return copy_host_to_device(begin, end, result, input_traversal, output_traversal,
            typename is_trivial_copy<InputIterator,OutputIterator>::type());
}


/////////////////
// Entry Point //
/////////////////

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy_host_to_device(InputIterator begin, 
                                     InputIterator end, 
                                     OutputIterator result)
{
    return copy_host_to_device(begin, end, result, 
            typename thrust::iterator_traversal<InputIterator>::type(),
            typename thrust::iterator_traversal<OutputIterator>::type());
}

} // end namespace device

} // end namespace detail

} // end namespace thrust

