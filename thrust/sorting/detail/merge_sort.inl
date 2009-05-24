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


/*! \file merge_sort.inl
 *  \brief Inline file for merge_sort.h.
 */

#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/sorting/detail/dispatch/merge_sort.h>

namespace thrust
{

namespace sorting
{

// Sorting Rules
// 1) Forward sorting methods to their stable versions when no unstable version exists
// 2) Use thrust::less<KeyType> as the default comparison method
// 3) Most general function dispatches based on iterator_category

/////////
// Key //
/////////

template<typename RandomAccessIterator>
  void merge_sort(RandomAccessIterator begin,
                  RandomAccessIterator end)
{
    typedef typename thrust::iterator_traits<RandomAccessIterator>::value_type KeyType;
    thrust::less<KeyType> comp;
    merge_sort(begin, end, comp);
}

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void merge_sort(RandomAccessIterator begin,
                  RandomAccessIterator end,
                  StrictWeakOrdering comp)
{
    stable_merge_sort(begin, end, comp);
}

template<typename RandomAccessIterator>
  void stable_merge_sort(RandomAccessIterator begin,
                         RandomAccessIterator end)
{
    typedef typename thrust::iterator_traits<RandomAccessIterator>::value_type KeyType;
    thrust::less<KeyType> comp;
    stable_merge_sort(begin, end, comp);
}

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_merge_sort(RandomAccessIterator begin,
                         RandomAccessIterator end,
                         StrictWeakOrdering comp)
{
    // Dispatch on iterator category
    thrust::sorting::detail::dispatch::stable_merge_sort(begin, end, comp,
    typename thrust::iterator_traits<RandomAccessIterator>::iterator_category());
}


///////////////
// Key Value //
///////////////

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator>
  void merge_sort_by_key(RandomAccessKeyIterator keys_begin,
                         RandomAccessKeyIterator keys_end,
                         RandomAccessValueIterator values_begin)
{
    typedef typename thrust::iterator_traits<RandomAccessKeyIterator>::value_type KeyType;
    thrust::less<KeyType> comp;
    merge_sort_by_key(keys_begin, keys_end, values_begin, comp);
}

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator,
         typename StrictWeakOrdering>
  void merge_sort_by_key(RandomAccessKeyIterator keys_begin,
                         RandomAccessKeyIterator keys_end,
                         RandomAccessValueIterator values_begin,
                         StrictWeakOrdering comp)
{
    stable_merge_sort_by_key(keys_begin, keys_end, values_begin, comp);
}


template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator>
  void stable_merge_sort_by_key(RandomAccessKeyIterator keys_begin,
                                RandomAccessKeyIterator keys_end,
                                RandomAccessValueIterator values_begin)
{
    typedef typename thrust::iterator_traits<RandomAccessKeyIterator>::value_type KeyType;
    thrust::less<KeyType> comp;
    stable_merge_sort_by_key(keys_begin, keys_end, values_begin, comp);
} 

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator,
         typename StrictWeakOrdering>
  void stable_merge_sort_by_key(RandomAccessKeyIterator keys_begin,
                                RandomAccessKeyIterator keys_end,
                                RandomAccessValueIterator values_begin,
                                StrictWeakOrdering comp)
{
    // dispatch on iterator category
    thrust::sorting::detail::dispatch::stable_merge_sort_by_key(keys_begin, keys_end, values_begin, comp,
            typename thrust::iterator_traits<RandomAccessKeyIterator>::iterator_category(),
            typename thrust::iterator_traits<RandomAccessValueIterator>::iterator_category());
}


} // end namespace sorting

} // end namespace thrust

