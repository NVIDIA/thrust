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
#include <thrust/copy.h>
#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/trivial_sequence.h>

#include <thrust/sorting/detail/dispatch/merge_sort.h>

namespace thrust
{

namespace sorting
{

// Sorting Rules
// 1) Forward sorting methods to their stable versions when no unstable version exists
// 2) Use thrust::less<KeyType> as the default comparison method
// 3) Most general variant of each function dispatches based on iterator_space

/////////
// Key //
/////////

template<typename RandomAccessIterator>
  void merge_sort(RandomAccessIterator first,
                  RandomAccessIterator last)
{
    typedef typename thrust::iterator_traits<RandomAccessIterator>::value_type KeyType;
    thrust::less<KeyType> comp;
    merge_sort(first, last, comp);
}

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void merge_sort(RandomAccessIterator first,
                  RandomAccessIterator last,
                  StrictWeakOrdering comp)
{
    stable_merge_sort(first, last, comp);
}

template<typename RandomAccessIterator>
  void stable_merge_sort(RandomAccessIterator first,
                         RandomAccessIterator last)
{
    typedef typename thrust::iterator_traits<RandomAccessIterator>::value_type KeyType;
    thrust::less<KeyType> comp;
    stable_merge_sort(first, last, comp);
}

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_merge_sort(RandomAccessIterator first,
                         RandomAccessIterator last,
                         StrictWeakOrdering comp)
{
    // ensure sequence has trivial iterators
    thrust::detail::trivial_sequence<RandomAccessIterator> keys(first, last);

    // Dispatch on iterator category
    thrust::sorting::detail::dispatch::stable_merge_sort(keys.begin(), keys.end(), comp,
            typename thrust::iterator_space<RandomAccessIterator>::type());

    // copy results back, if necessary
    if(!thrust::detail::is_trivial_iterator<RandomAccessIterator>::value)
        thrust::copy(keys.begin(), keys.end(), first);
}


///////////////
// Key Value //
///////////////

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void merge_sort_by_key(RandomAccessIterator1 keys_first,
                         RandomAccessIterator1 keys_last,
                         RandomAccessIterator2 values_first)
{
    typedef typename thrust::iterator_traits<RandomAccessIterator1>::value_type KeyType;
    thrust::less<KeyType> comp;
    merge_sort_by_key(keys_first, keys_last, values_first, comp);
}

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void merge_sort_by_key(RandomAccessIterator1 keys_first,
                         RandomAccessIterator1 keys_last,
                         RandomAccessIterator2 values_first,
                         StrictWeakOrdering comp)
{
    stable_merge_sort_by_key(keys_first, keys_last, values_first, comp);
}


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void stable_merge_sort_by_key(RandomAccessIterator1 keys_first,
                                RandomAccessIterator1 keys_last,
                                RandomAccessIterator2 values_first)
{
    typedef typename thrust::iterator_traits<RandomAccessIterator1>::value_type KeyType;
    thrust::less<KeyType> comp;
    stable_merge_sort_by_key(keys_first, keys_last, values_first, comp);
} 

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void stable_merge_sort_by_key(RandomAccessIterator1 keys_first,
                                RandomAccessIterator1 keys_last,
                                RandomAccessIterator2 values_first,
                                StrictWeakOrdering comp)
{
    // ensure sequences have trivial iterators
    RandomAccessIterator2 values_last = values_first + (keys_last - keys_first);
    thrust::detail::trivial_sequence<RandomAccessIterator1> keys(keys_first, keys_last);
    thrust::detail::trivial_sequence<RandomAccessIterator2> values(values_first, values_last);

    // dispatch on iterator category
    thrust::sorting::detail::dispatch::stable_merge_sort_by_key(keys.begin(), keys.end(), values.begin(), comp,
            typename thrust::iterator_space<RandomAccessIterator1>::type(),
            typename thrust::iterator_space<RandomAccessIterator2>::type());

    // copy results back, if necessary
    if(!thrust::detail::is_trivial_iterator<RandomAccessIterator1>::value)
        thrust::copy(keys.begin(), keys.end(), keys_first);
    if(!thrust::detail::is_trivial_iterator<RandomAccessIterator2>::value)
        thrust::copy(values.begin(), values.end(), values_first);
}


} // end namespace sorting

} // end namespace thrust

