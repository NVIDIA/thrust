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
#include <stdlib.h>           // for malloc & free

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/device/trivial_copy.h>
#include <thrust/detail/type_traits.h>

namespace thrust
{

namespace detail
{

namespace device
{

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::input_device_iterator_tag, 
                      thrust::output_host_iterator_tag)
{
    // XXX throw a compiler error here
}

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::random_access_device_iterator_tag, 
                      thrust::output_host_iterator_tag)
{
    typedef typename thrust::iterator_traits<InputIterator>::value_type T;

    // allocate temporary storage
    T *temp = reinterpret_cast<T*>(malloc(sizeof(T) * (end - begin)));
    T *temp_end = thrust::copy(begin, end, temp);

    result = std::copy(temp, temp_end, result);

    free(temp);
    return result;
}

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::random_access_device_iterator_tag, 
                      thrust::forward_host_iterator_tag)
{
    typedef typename thrust::iterator_traits<InputIterator>::value_type T;

    // allocate temporary storage
    T *temp = reinterpret_cast<T*>(malloc(sizeof(T) * (end - begin)));
    T *temp_end = thrust::copy(begin, end, temp);

    result = std::copy(temp, temp_end, result);

    free(temp);
    return result;
}

// const device pointer to host pointer with matching types
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::random_access_device_iterator_tag,
                      thrust::random_access_host_iterator_tag,
                      thrust::device_ptr<const typename thrust::iterator_traits<InputIterator>::value_type>, // match device_ptr<const T>
                      typename thrust::iterator_traits<InputIterator>::value_type *)                          // match T *
{
    // specialization for host to device when the types pointed to match (and operator= is not overloaded)

    // how many elements to copy?
    typename thrust::iterator_traits<OutputIterator>::difference_type n = end - begin;

    // what is the input type?
    typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;

    // XXX TODO check is_normal_iterator()
    trivial_copy_device_to_host(&*result, (&*begin).get(), n * sizeof(InputType));

    return result + n;
}


// device pointer to host pointer with matching types
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::random_access_device_iterator_tag,
                      thrust::random_access_host_iterator_tag,
                      thrust::device_ptr<typename thrust::iterator_traits<InputIterator>::value_type>, // match device_ptr<T>
                      typename thrust::iterator_traits<InputIterator>::value_type *)                    // match T *
{
    // use a typedef here so that old versions of gcc on OSX don't crash
    typedef typename thrust::device_ptr<const typename thrust::iterator_traits<InputIterator>::value_type> InputDevicePointer;

    return copy(begin, end, result,
            thrust::random_access_device_iterator_tag(),
            thrust::random_access_host_iterator_tag(),
            InputDevicePointer(),
            typename thrust::iterator_traits<OutputIterator>::pointer());
}

// random access device iterator to random access host iterator with mixed types
template<typename InputIterator,
         typename OutputIterator,
         typename InputIteratorPointer,
         typename OutputIteratorPointer>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::random_access_device_iterator_tag,
                      thrust::random_access_host_iterator_tag,
                      InputIteratorPointer,
                      OutputIteratorPointer)
{
    typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;

    // allocate temporary storage
    InputType *temp = reinterpret_cast<InputType*>(malloc(sizeof(InputType) * (end - begin)));
    InputType *temp_end = thrust::copy(begin, end, temp);

    result = thrust::copy(temp, temp_end, result);
    free(temp);

    return result;
}

// random access device iterator to random access host iterator
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::random_access_device_iterator_tag input_tag,
                      thrust::random_access_host_iterator_tag output_tag)
{
    // dispatch on the type of each iterator's pointers
    // XXX also need to dispatch on if output type fulfills has_trivial_assign_operator
    return copy(begin, end, result, input_tag, output_tag,
            typename thrust::iterator_traits<InputIterator>::pointer(),
            typename thrust::iterator_traits<OutputIterator>::pointer());
}


template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator begin,
                      InputIterator end,
                      OutputIterator result,
                      thrust::experimental::space::device,
                      thrust::experimental::space::host)
{
    return copy(begin, end, result, 
            typename thrust::iterator_traits<InputIterator>::iterator_category(),
            typename thrust::iterator_traits<OutputIterator>::iterator_category());
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
    return copy(begin, end, result, 
            typename thrust::iterator_traits<InputIterator>::iterator_category(),
            typename thrust::iterator_traits<OutputIterator>::iterator_category());
}

} // end namespace device

} // end namespace detail

} // end namespace thrust

