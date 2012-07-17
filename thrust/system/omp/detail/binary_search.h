/*
 *  Copyright 2008-2012 NVIDIA Corporation
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

#include <thrust/detail/config.h>
#include <thrust/system/omp/detail/tag.h>
#include <thrust/system/detail/generic/binary_search.h>

namespace thrust
{
namespace system
{
namespace omp
{
namespace detail
{


template <typename System, typename ForwardIterator, typename T, typename StrictWeakOrdering>
ForwardIterator lower_bound(dispatchable<System> &system,
                            ForwardIterator begin,
                            ForwardIterator end,
                            const T& value, 
                            StrictWeakOrdering comp)
{
    // omp prefers generic::lower_bound to cpp::lower_bound
    return thrust::system::detail::generic::lower_bound(system, begin, end, value, comp);
}


template <typename System, typename ForwardIterator, typename T, typename StrictWeakOrdering, typename Backend>
ForwardIterator upper_bound(dispatchable<System> &system,
                            ForwardIterator begin,
                            ForwardIterator end,
                            const T& value, 
                            StrictWeakOrdering comp)
{
    // omp prefers generic::upper_bound to cpp::upper_bound
    return thrust::system::detail::generic::upper_bound(system, begin, end, value, comp);
}


template <typename System, typename ForwardIterator, typename T, typename StrictWeakOrdering>
bool binary_search(dispatchable<System> &system,
                   ForwardIterator begin,
                   ForwardIterator end,
                   const T& value, 
                   StrictWeakOrdering comp)
{
    // omp prefers generic::binary_search to cpp::binary_search
    return thrust::system::detail::generic::binary_search(system, begin, end, value, comp);
}


} // end detail
} // end omp
} // end system
} // end thrust

