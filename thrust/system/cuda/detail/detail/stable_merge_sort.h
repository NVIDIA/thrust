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


/*! \file stable_merge_sort_dev.h
 *  \brief Defines the interface for a stable merge implementation on CUDA
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

template<typename System,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
void stable_merge_sort(dispatchable<System> &system,
                       RandomAccessIterator first,
                       RandomAccessIterator last,
                       StrictWeakOrdering comp);
    
template<typename System,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
void stable_merge_sort_by_key(dispatchable<System> &system,
                              RandomAccessIterator1 keys_begin,
                              RandomAccessIterator1 keys_end,
                              RandomAccessIterator2 values_begin,
                              StrictWeakOrdering comp);

} // end namespace detail
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

#include <thrust/system/cuda/detail/detail/stable_merge_sort.inl>

