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

#include <thrust/detail/type_traits.h>
#include <thrust/device_ptr.h>
#include <algorithm>
#include <thrust/detail/host/sort.h>

#include <thrust/iterator/detail/forced_iterator.h> // XXX remove this we we have a proper OMP sort

namespace thrust
{

namespace detail
{

namespace device
{

namespace omp
{

namespace dispatch
{

template<typename RandomAccessIterator, typename StrictWeakOrdering>
void stable_sort(RandomAccessIterator first, RandomAccessIterator last, StrictWeakOrdering comp,
                 thrust::detail::true_type)
{
  // RandomAccessIterator is trivial, so cast to a raw pointer and use std::stable_sort
  std::stable_sort(thrust::raw_pointer_cast(&*first),
                   thrust::raw_pointer_cast(&*last),
                   comp);
}

template<typename RandomAccessIterator, typename StrictWeakOrdering>
void stable_sort(RandomAccessIterator first, RandomAccessIterator last, StrictWeakOrdering comp,
                 thrust::detail::false_type)
{
  // RandomAccessIterator is not trivial, so use host's stable_sort implementation
  thrust::detail::host::stable_sort(thrust::detail::make_forced_iterator(first, thrust::host_space_tag()),
                                    thrust::detail::make_forced_iterator(last,  thrust::host_space_tag()),
                                    comp);
}

} // end dispatch

} // end omp

} // end device

} // end detail

} // end thrust

