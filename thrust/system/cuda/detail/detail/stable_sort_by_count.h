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
#include <thrust/system/cuda/detail/tag.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace detail
{


template<unsigned int count,
         typename System,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename Compare>
void stable_sort_by_count(dispatchable<System> &system,
                          RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first,
                          Compare comp);


} // end detail
} // end detail
} // end cuda
} // end system
} // end thrust

#include <thrust/system/cuda/detail/detail/stable_sort_by_count.inl>

