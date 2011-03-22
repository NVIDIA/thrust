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


/*! \file binary_search.h
 *  \brief Dispatch layer of the binary search functions.
 */

#pragma once

#include <thrust/iterator/iterator_categories.h>

#include <thrust/detail/host/binary_search.h>
#include <thrust/detail/device/binary_search.h>

namespace thrust
{

namespace detail
{

namespace dispatch
{

////////////////
// Host Paths //
////////////////
template <class ForwardIterator, class T, class StrictWeakOrdering>
ForwardIterator lower_bound(ForwardIterator begin,
                            ForwardIterator end,
                            const T& value, 
                            StrictWeakOrdering comp,
                            thrust::host_space_tag)
{
    return thrust::detail::host::lower_bound(begin, end, value, comp);
}

template <class ForwardIterator, class InputIterator, class OutputIterator, class StrictWeakOrdering>
OutputIterator lower_bound(ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output,
                           StrictWeakOrdering comp,
                           thrust::host_space_tag,
                           thrust::host_space_tag,
                           thrust::host_space_tag)
{
    return thrust::detail::host::lower_bound(begin, end, values_begin, values_end, output, comp);
}

template <class ForwardIterator, class T, class StrictWeakOrdering>
ForwardIterator upper_bound(ForwardIterator begin,
                            ForwardIterator end,
                            const T& value, 
                            StrictWeakOrdering comp,
                            thrust::host_space_tag)
{
    return thrust::detail::host::upper_bound(begin, end, value, comp);
}

template <class ForwardIterator, class InputIterator, class OutputIterator, class StrictWeakOrdering>
OutputIterator upper_bound(ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output,
                           StrictWeakOrdering comp,
                           thrust::host_space_tag,
                           thrust::host_space_tag,
                           thrust::host_space_tag)
{
    return thrust::detail::host::upper_bound(begin, end, values_begin, values_end, output, comp);
}

template <class ForwardIterator, class T, class StrictWeakOrdering>
bool binary_search(ForwardIterator begin,
                   ForwardIterator end,
                   const T& value, 
                   StrictWeakOrdering comp,
                   thrust::host_space_tag)
{
    return thrust::detail::host::binary_search(begin, end, value, comp);
}

template <class ForwardIterator, class InputIterator, class OutputIterator, class StrictWeakOrdering>
OutputIterator binary_search(ForwardIterator begin, 
                             ForwardIterator end,
                             InputIterator values_begin, 
                             InputIterator values_end,
                             OutputIterator output,
                             StrictWeakOrdering comp,
                             thrust::host_space_tag,
                             thrust::host_space_tag,
                             thrust::host_space_tag)
{
    return thrust::detail::host::binary_search(begin, end, values_begin, values_end, output, comp);
}

//////////////////
// Device Paths //
//////////////////
template <class ForwardIterator, class T, class StrictWeakOrdering>
ForwardIterator lower_bound(ForwardIterator begin,
                            ForwardIterator end,
                            const T& value, 
                            StrictWeakOrdering comp,
                            thrust::device_space_tag)
{
    return thrust::detail::device::lower_bound(begin, end, value, comp);
}

template <class ForwardIterator, class InputIterator, class OutputIterator, class StrictWeakOrdering>
OutputIterator lower_bound(ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output,
                           StrictWeakOrdering comp,
                           thrust::device_space_tag,
                           thrust::device_space_tag,
                           thrust::device_space_tag)
{
    return thrust::detail::device::lower_bound(begin, end, values_begin, values_end, output, comp);
}

template <class ForwardIterator, class T, class StrictWeakOrdering>
ForwardIterator upper_bound(ForwardIterator begin,
                            ForwardIterator end,
                            const T& value, 
                            StrictWeakOrdering comp,
                            thrust::device_space_tag)
{
    return thrust::detail::device::upper_bound(begin, end, value, comp);
}

template <class ForwardIterator, class InputIterator, class OutputIterator, class StrictWeakOrdering>
OutputIterator upper_bound(ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output,
                           StrictWeakOrdering comp,
                           thrust::device_space_tag,
                           thrust::device_space_tag,
                           thrust::device_space_tag)
{
    return thrust::detail::device::upper_bound(begin, end, values_begin, values_end, output, comp);
}

template <class ForwardIterator, class T, class StrictWeakOrdering>
bool binary_search(ForwardIterator begin,
                   ForwardIterator end,
                   const T& value, 
                   StrictWeakOrdering comp,
                   thrust::device_space_tag)
{
    return thrust::detail::device::binary_search(begin, end, value, comp);
}

template <class ForwardIterator, class InputIterator, class OutputIterator, class StrictWeakOrdering>
OutputIterator binary_search(ForwardIterator begin, 
                             ForwardIterator end,
                             InputIterator values_begin, 
                             InputIterator values_end,
                             OutputIterator output,
                             StrictWeakOrdering comp,
                             thrust::device_space_tag,
                             thrust::device_space_tag,
                             thrust::device_space_tag)
{
    return thrust::detail::device::binary_search(begin, end, values_begin, values_end, output, comp);
}

} // end dispatch

} // end detail

} // end thrust

