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


/*! \file sort.inl
 *  \brief Inline file for sort.h.
 */

#include <thrust/sort.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/functional.h>
#include <thrust/detail/dispatch/sort.h>

namespace thrust
{

///////////////////////
// Sort Entry Points //
///////////////////////

template<typename RandomAccessIterator>
  void sort(RandomAccessIterator first,
            RandomAccessIterator last)
{
    // dispatch on iterator category
    thrust::detail::dispatch::sort(first, last,
            typename thrust::iterator_traits<RandomAccessIterator>::iterator_category());
}

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void sort(RandomAccessIterator first,
            RandomAccessIterator last,
            StrictWeakOrdering comp)
{
    // dispatch on iterator category
    thrust::detail::dispatch::sort(first, last, comp,
            typename thrust::iterator_traits<RandomAccessIterator>::iterator_category());
}

template<typename RandomAccessIterator>
  void stable_sort(RandomAccessIterator first,
                   RandomAccessIterator last)
{
    // dispatch on iterator category
    thrust::detail::dispatch::stable_sort(first, last,
            typename thrust::iterator_traits<RandomAccessIterator>::iterator_category());
} 

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_sort(RandomAccessIterator first,
                   RandomAccessIterator last,
                   StrictWeakOrdering comp)
{
    // dispatch on iterator category
    thrust::detail::dispatch::stable_sort(first, last, comp,
            typename thrust::iterator_traits<RandomAccessIterator>::iterator_category());
}




// The STL has no analog of sort_by_key() or stable_sort_by_key(),
// so we rely on merge_sort_by_key() and radix_sort_by_key()

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator>
  void sort_by_key(RandomAccessKeyIterator keys_first,
                   RandomAccessKeyIterator keys_last,
                   RandomAccessValueIterator values_first)
{
    stable_sort_by_key(keys_first, keys_last, values_first);
}

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator,
         typename StrictWeakOrdering>
  void sort_by_key(RandomAccessKeyIterator keys_first,
                   RandomAccessKeyIterator keys_last,
                   RandomAccessValueIterator values_first,
                   StrictWeakOrdering comp)
{
    stable_sort_by_key(keys_first, keys_last, values_first, comp);
}

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator>
  void stable_sort_by_key(RandomAccessKeyIterator keys_first,
                          RandomAccessKeyIterator keys_last,
                          RandomAccessValueIterator values_first)
{
    // dispatch on iterator category
    thrust::detail::dispatch::stable_sort_by_key(keys_first, keys_last, values_first,
            typename thrust::iterator_traits<RandomAccessKeyIterator>::iterator_category(),
            typename thrust::iterator_traits<RandomAccessValueIterator>::iterator_category());
}

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator,
         typename StrictWeakOrdering>
  void stable_sort_by_key(RandomAccessKeyIterator keys_first,
                          RandomAccessKeyIterator keys_last,
                          RandomAccessValueIterator values_first,
                          StrictWeakOrdering comp)
{
    // dispatch on iterator category
    thrust::detail::dispatch::stable_sort_by_key(keys_first, keys_last, values_first, comp,
            typename thrust::iterator_traits<RandomAccessKeyIterator>::iterator_category(),
            typename thrust::iterator_traits<RandomAccessValueIterator>::iterator_category());
}

} // last namespace thrust

