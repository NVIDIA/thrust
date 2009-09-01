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
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/raw_buffer.h>

#include <thrust/detail/device/trivial_copy.h>

namespace thrust
{

namespace detail
{

namespace device
{

namespace detail
{

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy_device_to_device(InputIterator begin, 
                                       InputIterator end, 
                                       OutputIterator result,
                                       thrust::detail::false_type)
{
    // general case (mixed types)
    typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;

#ifdef __CUDACC__
    return thrust::transform(begin, end, result, thrust::identity<InputType>());
#else
    // we're not compiling with nvcc: copy [begin, end) to temp host memory
    typename thrust::iterator_traits<InputIterator>::difference_type n = thrust::distance(begin, end);

    raw_buffer<InputType, host_space_tag> temp1(begin, end);

    // transform temp1 to OutputType in host memory
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;
    raw_buffer<OutputType, host_space_tag> temp2(n);
    thrust::transform(temp1.begin(), temp1.end(), temp2.begin(), thrust::identity<InputType>());

    // copy temp2 to device
    result = thrust::copy(temp2.begin(), temp2.end(), result);

    return result;
#endif // __CUDACC__
}


template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy_device_to_device(InputIterator begin, 
                                       InputIterator end, 
                                       OutputIterator result,
                                       thrust::detail::true_type)
{
    // specialization for device to device when the value_types match, operator= is not overloaded,
    // and the iterators are pointers

    // how many elements to copy?
    typename thrust::iterator_traits<OutputIterator>::difference_type n = end - begin;

    // figure out the type of the element to copy
    typedef typename thrust::iterator_traits<OutputIterator>::value_type T;

    // XXX TODO use raw_pointer_cast
    void * dest       = thrust::raw_pointer_cast(&*result);
    const void * src  = thrust::raw_pointer_cast(&*begin);

    thrust::detail::device::trivial_copy_device_to_device(dest, src, n * sizeof(T));

    return result + n;
}

} // end namespace detail

/////////////////
// Entry Point //
/////////////////

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy_device_to_device(InputIterator begin, 
                                       InputIterator end, 
                                       OutputIterator result)
{
    typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

    const bool use_trivial_copy = 
        thrust::detail::is_same<InputType, OutputType>::value
        && thrust::detail::is_trivial_iterator<InputIterator>::value 
        && thrust::detail::is_trivial_iterator<OutputIterator>::value;

    return detail::copy_device_to_device(begin, end, result,
            thrust::detail::integral_constant<bool, use_trivial_copy>());

}

} // end namespace device

} // end namespace detail

} // end namespace thrust

