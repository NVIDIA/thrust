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


/*! \file stable_radix_sort.h
 *  \brief Device implementation of stable_radix_sort
 */

#pragma once

#include <thrust/sorting/detail/device/cuda/stable_radix_sort.h>

namespace thrust
{

namespace sorting
{

namespace detail
{

namespace device
{

template<typename RandomAccessIterator>
void stable_radix_sort(RandomAccessIterator first,
                       RandomAccessIterator last)
{
    thrust::sorting::detail::device::cuda::stable_radix_sort(first, last);
}

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
void stable_radix_sort_by_key(RandomAccessIterator1 keys_first,
                              RandomAccessIterator1 keys_last,
                              RandomAccessIterator2 values_first)
{
    thrust::sorting::detail::device::cuda::stable_radix_sort_by_key(keys_first, keys_last, values_first);
}

} // end namespace device

} // end namespace detail

} // end namespace sorting

} // end namespace thrust

