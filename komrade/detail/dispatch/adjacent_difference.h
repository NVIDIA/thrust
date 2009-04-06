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


/*! \file adjacent_difference.h
 *  \brief Defines the interface to the dispatch
 *         layer of the adjacent_difference function.
 */

#pragma once

#include <numeric>

#include <komrade/adjacent_difference.h>
#include <komrade/device_ptr.h>
#include <komrade/device_malloc.h>
#include <komrade/device_free.h>
#include <komrade/copy.h>
#include <komrade/transform.h>

namespace komrade
{

namespace detail
{

namespace dispatch
{

///////////////
// Host Path //
///////////////
template <class InputIterator, class OutputIterator, class BinaryFunction>
OutputIterator adjacent_difference(InputIterator first, InputIterator last,
                                   OutputIterator result,
                                   BinaryFunction binary_op,
                                   komrade::input_host_iterator_tag)
{
    return std::adjacent_difference(first, last, result, binary_op);
}


namespace detail
{

// XXX generalize this and put it in komrade:: namespace
// consider defining disjoint_ranges() instead
template <class DeviceIterator1, class DeviceIterator2, class Type1, class Type2>
bool within_range(DeviceIterator1 first, DeviceIterator1 last, DeviceIterator2 pt,
                  komrade::device_ptr<Type1>, komrade::device_ptr<Type2>)
{
    const void * first_ptr = komrade::raw_pointer_cast(&*first); 
    const void * last_ptr  = komrade::raw_pointer_cast(&*last);
    const void * pt_ptr    = komrade::raw_pointer_cast(&*pt);
    return pt_ptr >= first_ptr && pt_ptr < last_ptr;
}

template <class DeviceIterator1, class DeviceIterator2, class DeviceIterator1Pointer, class DeviceIterator2Pointer>
bool within_range(DeviceIterator1 first, DeviceIterator1 last, DeviceIterator2 pt,
                  DeviceIterator1Pointer, DeviceIterator2Pointer)
{
    return false;
}

template <class DeviceIterator1, class DeviceIterator2>
bool within_range(DeviceIterator1 first, DeviceIterator1 last, DeviceIterator2 pt)
{
    return within_range(first, last, pt,
            typename komrade::iterator_traits<DeviceIterator1>::pointer(),
            typename komrade::iterator_traits<DeviceIterator2>::pointer());
}

} // end namespace detail

/////////////////
// Device Path //
/////////////////
template <class InputIterator, class OutputIterator, class BinaryFunction>
OutputIterator adjacent_difference(InputIterator first, InputIterator last,
                                   OutputIterator result,
                                   BinaryFunction binary_op,
                                   komrade::random_access_device_iterator_tag)
{
    typedef typename komrade::iterator_traits<InputIterator>::value_type InputType;

    if(first == last)
    {
        // empty range, nothing to do
        return result; 
    }
    else if(detail::within_range(first, last, result))
    {
        // an in-place operation is requested, copy the input and call the entry point
        // XXX a special-purpose kernel would be faster here since
        // only block boundaries need to be copied
        komrade::device_ptr<InputType> input_copy = komrade::device_malloc<InputType>(last - first);
        komrade::copy(first, last, input_copy);
        komrade::adjacent_difference(input_copy, input_copy + (last - first), result, binary_op);
        komrade::device_free(input_copy);
    }
    else
    {
        // XXX a special-purpose kernel would be faster here
        *result = *first;
        komrade::transform(first + 1, last, first, result + 1, binary_op); 
    } // end else

    return result + (last - first);
}

} // end dispatch

} // end detail

} // end komrade

