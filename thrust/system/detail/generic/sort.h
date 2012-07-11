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
#include <thrust/system/detail/generic/tag.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename System,
         typename RandomAccessIterator>
  void sort(thrust::dispatchable<System> &system,
            RandomAccessIterator first,
            RandomAccessIterator last);


template<typename System,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void sort(thrust::dispatchable<System> &system,
            RandomAccessIterator first,
            RandomAccessIterator last,
            StrictWeakOrdering comp);


template<typename System,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void sort_by_key(thrust::dispatchable<System> &system,
                   RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first);


template<typename System,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void sort_by_key(thrust::dispatchable<System> &system,
                   RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first,
                   StrictWeakOrdering comp);


template<typename System,
         typename RandomAccessIterator>
  void stable_sort(thrust::dispatchable<System> &system,
                   RandomAccessIterator first,
                   RandomAccessIterator last);


// XXX it is an error to call this function; it has no implementation
template<typename System,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_sort(thrust::dispatchable<System> &system,
                   RandomAccessIterator first,
                   RandomAccessIterator last,
                   StrictWeakOrdering comp);


template<typename System,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void stable_sort_by_key(thrust::dispatchable<System> &system,
                          RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first);


// XXX it is an error to call this function; it has no implementation
template<typename System,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void stable_sort_by_key(thrust::dispatchable<System> &system,
                          RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first,
                          StrictWeakOrdering comp);


template<typename System, typename ForwardIterator>
  bool is_sorted(thrust::dispatchable<System> &system,
                 ForwardIterator first,
                 ForwardIterator last);


template<typename System,
         typename ForwardIterator,
         typename Compare>
  bool is_sorted(thrust::dispatchable<System> &system,
                 ForwardIterator first,
                 ForwardIterator last,
                 Compare comp);


template<typename System, typename ForwardIterator>
  ForwardIterator is_sorted_until(thrust::dispatchable<System> &system,
                                  ForwardIterator first,
                                  ForwardIterator last);


template<typename System,
         typename ForwardIterator,
         typename Compare>
  ForwardIterator is_sorted_until(thrust::dispatchable<System> &system,
                                  ForwardIterator first,
                                  ForwardIterator last,
                                  Compare comp);


} // end generic
} // end detail
} // end system
} // end thrust

#include <thrust/system/detail/generic/sort.inl>

