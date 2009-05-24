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

#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/sorting/detail/dispatch/radix_sort.h>

namespace thrust
{

namespace sorting
{

/////////
// Key //
/////////

template<typename RandomAccessIterator>
  void radix_sort(RandomAccessIterator begin,
                  RandomAccessIterator end)
{
    stable_radix_sort(begin, end);
}

template<typename RandomAccessIterator>
  void stable_radix_sort(RandomAccessIterator begin,
                         RandomAccessIterator end)
{
    // dispatch on iterator category
    thrust::sorting::detail::dispatch::stable_radix_sort(begin, end,
            typename thrust::iterator_traits<RandomAccessIterator>::iterator_category());
}


///////////////
// Key Value //
///////////////

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void radix_sort_by_key(RandomAccessIterator1 keys_begin,
                     RandomAccessIterator1 keys_end,
                     RandomAccessIterator2 values_begin)
{
    stable_radix_sort_by_key(keys_begin, keys_end, values_begin);
}


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void stable_radix_sort_by_key(RandomAccessIterator1 keys_begin,
                            RandomAccessIterator1 keys_end,
                            RandomAccessIterator2 values_begin)
{
    // dispatch on iterator category
    thrust::sorting::detail::dispatch::stable_radix_sort_by_key(keys_begin, keys_end, values_begin,
            typename thrust::iterator_traits<RandomAccessIterator1>::iterator_category(),
            typename thrust::iterator_traits<RandomAccessIterator2>::iterator_category());
}


} // end namespace sorting

} // end namespace thrust

