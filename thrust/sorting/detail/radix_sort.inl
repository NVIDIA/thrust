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


/*! \file radix_sort.inl
 *  \brief Inline file for radix_sort.h.
 */

#include <thrust/copy.h>
#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/trivial_sequence.h>

#include <thrust/sorting/detail/dispatch/radix_sort.h>

namespace thrust
{

namespace sorting
{

/////////
// Key //
/////////

template<typename RandomAccessIterator>
  void radix_sort(RandomAccessIterator first,
                  RandomAccessIterator last)
{
    stable_radix_sort(first, last);
}

template<typename RandomAccessIterator>
  void stable_radix_sort(RandomAccessIterator first,
                         RandomAccessIterator last)
{
    // ensure key sequence has trivial iterators
    thrust::detail::trivial_sequence<RandomAccessIterator> keys(first, last);

    // dispatch on iterator category
    thrust::sorting::detail::dispatch::stable_radix_sort(keys.begin(), keys.end(),
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
  void radix_sort_by_key(RandomAccessIterator1 keys_first,
                         RandomAccessIterator1 keys_last,
                         RandomAccessIterator2 values_first)
{
    stable_radix_sort_by_key(keys_first, keys_last, values_first);
}


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void stable_radix_sort_by_key(RandomAccessIterator1 keys_first,
                                RandomAccessIterator1 keys_last,
                                RandomAccessIterator2 values_first)
{
    // ensure key sequence has trivial iterators
    thrust::detail::trivial_sequence<RandomAccessIterator1> keys(keys_first, keys_last);

    // dispatch on iterator category
    thrust::sorting::detail::dispatch::stable_radix_sort_by_key(keys.begin(), keys.end(), values_first,
            typename thrust::iterator_space<RandomAccessIterator1>::type(),
            typename thrust::iterator_space<RandomAccessIterator2>::type());

    // copy results back, if necessary
    if(!thrust::detail::is_trivial_iterator<RandomAccessIterator1>::value)
        thrust::copy(keys.begin(), keys.end(), keys_first);
}


} // end namespace sorting

} // end namespace thrust

