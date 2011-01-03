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


/*! \file stable_merge_sort_dev.h
 *  \brief Defines the interface for a stable merge implementation on CUDA
 */

#pragma once

namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{
namespace detail
{

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
void stable_merge_sort(RandomAccessIterator first,
                       RandomAccessIterator last,
                       StrictWeakOrdering comp);
    
template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
void stable_merge_sort_by_key(RandomAccessIterator1 keys_begin,
                              RandomAccessIterator1 keys_end,
                              RandomAccessIterator2 values_begin,
                              StrictWeakOrdering comp);

} // end namespace detail
} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

#include "stable_merge_sort.inl"

