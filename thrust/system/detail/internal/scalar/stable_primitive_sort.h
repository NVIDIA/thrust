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

namespace thrust
{
namespace system
{
namespace detail
{
namespace internal
{
namespace scalar
{

template<typename RandomAccessIterator>
void stable_primitive_sort(RandomAccessIterator first,
                           RandomAccessIterator last);

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
void stable_primitive_sort_by_key(RandomAccessIterator1 keys_first,
                                  RandomAccessIterator1 keys_last,
                                  RandomAccessIterator2 values_first);

} // end namespace scalar
} // end namespace internal
} // end namespace detail
} // end namespace system
} // end namespace thrust

#include <thrust/system/detail/internal/scalar/stable_primitive_sort.inl>

