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


#include <thrust/iterator/iterator_traits.h>

#include <thrust/device_ptr.h>
#include <thrust/detail/raw_buffer.h>
#include <thrust/copy.h>
#include <thrust/transform.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace generic
{
namespace detail
{

// XXX generalize this and put it in thrust:: namespace
// consider defining disjoint_ranges() instead
template <class DeviceIterator1, class DeviceIterator2, class Type1, class Type2>
bool within_range(DeviceIterator1 first, DeviceIterator1 last, DeviceIterator2 pt,
                  thrust::device_ptr<Type1>, thrust::device_ptr<Type2>)
{
    const void * first_ptr = thrust::raw_pointer_cast(&*first); 
    const void * last_ptr  = thrust::raw_pointer_cast(&*last);
    const void * pt_ptr    = thrust::raw_pointer_cast(&*pt);
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
            typename thrust::iterator_traits<DeviceIterator1>::pointer(),
            typename thrust::iterator_traits<DeviceIterator2>::pointer());
}

} // end namespace detail



template <class InputIterator, class OutputIterator, class BinaryFunction>
OutputIterator adjacent_difference(InputIterator first, InputIterator last,
                                   OutputIterator result,
                                   BinaryFunction binary_op)
{
    typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;
    typedef typename thrust::iterator_space<InputIterator>::type Space;

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
        thrust::detail::raw_buffer<InputType, Space> input_copy(first, last);
        thrust::detail::device::generic::adjacent_difference(input_copy.begin(), input_copy.end(), result, binary_op);
    }
    else
    {
        // XXX a special-purpose kernel would be faster here
        *result = *first;
        thrust::transform(first + 1, last, first, result + 1, binary_op); 
    } // end else

    return result + (last - first);
}

} // end namespace generic
} // end namespace device
} // end namespace detail
} // end namespace thrust

