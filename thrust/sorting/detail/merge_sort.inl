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
#include <thrust/detail/trivial_sequence.h>

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
    // ensure sequence has trivial iterators
    thrust::detail::trivial_sequence<RandomAccessIterator> keys(begin, end);

    // Dispatch on iterator category
    thrust::sorting::detail::dispatch::stable_merge_sort(keys.begin(), keys.end(), comp,
            typename thrust::iterator_space<RandomAccessIterator>::type());

    // copy results back, if necessary
    if(!thrust::detail::is_trivial_iterator<RandomAccessIterator>::value)
        thrust::copy(keys.begin(), keys.end(), begin);
}


///////////////
// Key Value //
///////////////

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void merge_sort_by_key(RandomAccessIterator1 keys_begin,
                         RandomAccessIterator1 keys_end,
                         RandomAccessIterator2 values_begin)
{
    typedef typename thrust::iterator_traits<RandomAccessIterator1>::value_type KeyType;
    thrust::less<KeyType> comp;
    merge_sort_by_key(keys_begin, keys_end, values_begin, comp);
}

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void merge_sort_by_key(RandomAccessIterator1 keys_begin,
                         RandomAccessIterator1 keys_end,
                         RandomAccessIterator2 values_begin,
                         StrictWeakOrdering comp)
{
    stable_merge_sort_by_key(keys_begin, keys_end, values_begin, comp);
}


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void stable_merge_sort_by_key(RandomAccessIterator1 keys_begin,
                                RandomAccessIterator1 keys_end,
                                RandomAccessIterator2 values_begin)
{
    typedef typename thrust::iterator_traits<RandomAccessIterator1>::value_type KeyType;
    thrust::less<KeyType> comp;
    stable_merge_sort_by_key(keys_begin, keys_end, values_begin, comp);
} 

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void stable_merge_sort_by_key(RandomAccessIterator1 keys_begin,
                                RandomAccessIterator1 keys_end,
                                RandomAccessIterator2 values_begin,
                                StrictWeakOrdering comp)
{
    // ensure sequences have trivial iterators
    RandomAccessIterator2 values_end = values_begin + (keys_end - keys_begin);
    thrust::detail::trivial_sequence<RandomAccessIterator1> keys(keys_begin, keys_end);
    thrust::detail::trivial_sequence<RandomAccessIterator2> values(values_begin, values_end);

    // dispatch on iterator category
    thrust::sorting::detail::dispatch::stable_merge_sort_by_key(keys.begin(), keys.end(), values.begin(), comp,
            typename thrust::iterator_space<RandomAccessIterator1>::type(),
            typename thrust::iterator_space<RandomAccessIterator2>::type());

    // copy results back, if necessary
    if(!thrust::detail::is_trivial_iterator<RandomAccessIterator1>::value)
        thrust::copy(keys.begin(), keys.end(), keys_begin);
    if(!thrust::detail::is_trivial_iterator<RandomAccessIterator2>::value)
        thrust::copy(values.begin(), values.end(), values_begin);
}


} // end namespace sorting

} // end namespace thrust

