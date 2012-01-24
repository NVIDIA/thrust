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

#include <thrust/reverse.h>

#include <thrust/detail/type_traits.h>
#include <thrust/iterator/iterator_traits.h>

#include <thrust/system/detail/internal/scalar/stable_merge_sort.h>
#include <thrust/system/detail/internal/scalar/stable_radix_sort.h>

namespace thrust
{
namespace system
{
namespace cpp
{
namespace detail
{
namespace dispatch
{

////////////////
// Radix Sort //
////////////////

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
void stable_sort(RandomAccessIterator first,
                 RandomAccessIterator last,
                 StrictWeakOrdering comp,
                 thrust::detail::true_type)
{
  thrust::system::detail::internal::scalar::stable_radix_sort(first, last);
        
  // if comp is greater<T> then reverse the keys
  typedef typename thrust::iterator_traits<RandomAccessIterator>::value_type KeyType;
  const static bool reverse = thrust::detail::is_same<StrictWeakOrdering, typename thrust::greater<KeyType> >::value;

  if (reverse)
    thrust::reverse(first, last);
}

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
void stable_sort_by_key(RandomAccessIterator1 first1,
                        RandomAccessIterator1 last1,
                        RandomAccessIterator2 first2,
                        StrictWeakOrdering comp,
                        thrust::detail::true_type)
{
  // if comp is greater<T> then reverse the keys and values
  typedef typename thrust::iterator_traits<RandomAccessIterator1>::value_type KeyType;
  const static bool reverse = thrust::detail::is_same<StrictWeakOrdering, typename thrust::greater<KeyType> >::value;

  // note, we also have to reverse the (unordered) input to preserve stability
  if (reverse)
  {
    thrust::reverse(first1,  last1);
    thrust::reverse(first2, first2 + (last1 - first1));
  }

  thrust::system::detail::internal::scalar::stable_radix_sort_by_key(first1, last1, first2);

  if (reverse)
  {
    thrust::reverse(first1,  last1);
    thrust::reverse(first2, first2 + (last1 - first1));
  }
}

////////////////
// Merge Sort //
////////////////

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
void stable_sort(RandomAccessIterator first,
                 RandomAccessIterator last,
                 StrictWeakOrdering comp,
                 thrust::detail::false_type)
{
  thrust::system::detail::internal::scalar::stable_merge_sort(first, last, comp);
}

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
void stable_sort_by_key(RandomAccessIterator1 first1,
                        RandomAccessIterator1 last1,
                        RandomAccessIterator2 first2,
                        StrictWeakOrdering comp,
                        thrust::detail::false_type)
{
  thrust::system::detail::internal::scalar::stable_merge_sort_by_key(first1, last1, first2, comp);
}

} // end namespace dispatch
} // end namespace detail
} // end namespace cpp
} // end namespace system
} // end namespace thrust

