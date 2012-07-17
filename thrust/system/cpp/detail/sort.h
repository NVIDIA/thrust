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
#include <thrust/system/cpp/detail/tag.h>
#include <thrust/system/detail/internal/scalar/sort.h>

namespace thrust
{
namespace system
{
namespace cpp
{
namespace detail
{

template<typename System,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_sort(dispatchable<System> &,
                   RandomAccessIterator first,
                   RandomAccessIterator last,
                   StrictWeakOrdering comp)
{
  thrust::system::detail::internal::scalar::stable_sort(first, last, comp);
}

template<typename System,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void stable_sort_by_key(dispatchable<System> &,
                          RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first,
                          StrictWeakOrdering comp)
{
  thrust::system::detail::internal::scalar::stable_sort_by_key(keys_first, keys_last, values_first, comp);
}

} // end namespace detail
} // end namespace cpp
} // end namespace system
} // end namespace thrust

