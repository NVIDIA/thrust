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

#include <thrust/detail/config.h>
#include <thrust/detail/backend/generic/sort.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace generic
{


template<typename RandomAccessIterator>
  void sort(tag,
            RandomAccessIterator first,
            RandomAccessIterator last)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type; 
  thrust::sort(first, last, thrust::less<value_type>());
} // end sort()


template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void sort(tag,
            RandomAccessIterator first,
            RandomAccessIterator last,
            StrictWeakOrdering comp)
{
  // implement with stable_sort
  thrust::stable_sort(first,last,comp);
} // end sort()


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void sort_by_key(tag,
                   RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type;
  thrust::sort_by_key(keys_first, keys_last, values_first, thrust::less<value_type>());
} // end sort_by_key()


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void sort_by_key(tag,
                   RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first,
                   StrictWeakOrdering comp)
{
  // implement with stable_sort_by_key
  thrust::stable_sort_by_key(keys_first, keys_last, values_first, comp);
} // end sort_by_key()


template<typename RandomAccessIterator>
  void stable_sort(tag,
                   RandomAccessIterator first,
                   RandomAccessIterator last)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;
  thrust::stable_sort(first, last, thrust::less<value_type>());
} // end stable_sort()


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void stable_sort_by_key(tag,
                          RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first)
{
  typedef typename iterator_value<RandomAccessIterator1>::type value_type;
  thrust::stable_sort_by_key(keys_first, keys_last, values_first, thrust::less<value_type>());
} // end stable_sort_by_key()


} // end generic
} // end backend
} // end detail
} // end thrust

