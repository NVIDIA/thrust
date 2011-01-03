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


/*! \file sort.inl
 *  \brief Inline file for sort.h.
 */

#include <thrust/sort.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/detail/dispatch/sort.h>

namespace thrust
{

///////////////
// Key Sorts //
///////////////

template<typename RandomAccessIterator>
  void sort(RandomAccessIterator first,
            RandomAccessIterator last)
{
    typedef typename thrust::iterator_traits<RandomAccessIterator>::value_type KeyType;

    // default comparison method is less<KeyType>
    thrust::sort(first, last, thrust::less<KeyType>());
}

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void sort(RandomAccessIterator first,
            RandomAccessIterator last,
            StrictWeakOrdering comp)
{
    // dispatch on space
    thrust::detail::dispatch::sort(first, last, comp,
            typename thrust::iterator_space<RandomAccessIterator>::type());
}

template<typename RandomAccessIterator>
  void stable_sort(RandomAccessIterator first,
                   RandomAccessIterator last)
{
    typedef typename thrust::iterator_traits<RandomAccessIterator>::value_type KeyType;

    // default comparison method is less<KeyType>
    thrust::stable_sort(first, last, thrust::less<KeyType>());
} 

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_sort(RandomAccessIterator first,
                   RandomAccessIterator last,
                   StrictWeakOrdering comp)
{
    // dispatch on space
    thrust::detail::dispatch::stable_sort(first, last, comp,
            typename thrust::iterator_space<RandomAccessIterator>::type());
}



/////////////////////
// Key-Value Sorts //
/////////////////////

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void sort_by_key(RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first)
{
    typedef typename thrust::iterator_traits<RandomAccessIterator1>::value_type KeyType;

    // default comparison method is less<KeyType>
    sort_by_key(keys_first, keys_last, values_first, thrust::less<KeyType>());
}

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void sort_by_key(RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first,
                   StrictWeakOrdering comp)
{
    // dispatch on space
    thrust::detail::dispatch::sort_by_key(keys_first, keys_last, values_first, comp,
            typename thrust::iterator_space<RandomAccessIterator1>::type(),
            typename thrust::iterator_space<RandomAccessIterator2>::type());
}

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void stable_sort_by_key(RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first)
{
    typedef typename thrust::iterator_traits<RandomAccessIterator1>::value_type KeyType;

    // default comparison method is less<KeyType>
    thrust::stable_sort_by_key(keys_first, keys_last, values_first, thrust::less<KeyType>());
}

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void stable_sort_by_key(RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first,
                          StrictWeakOrdering comp)
{
    // dispatch on space
    thrust::detail::dispatch::stable_sort_by_key(keys_first, keys_last, values_first, comp,
            typename thrust::iterator_space<RandomAccessIterator1>::type(),
            typename thrust::iterator_space<RandomAccessIterator2>::type());
}

template<typename ForwardIterator>
  bool is_sorted(ForwardIterator first,
                 ForwardIterator last)
{
  return thrust::is_sorted_until(first, last) == last;
} // end is_sorted()

template<typename ForwardIterator,
         typename Compare>
  bool is_sorted(ForwardIterator first,
                 ForwardIterator last,
                 Compare comp)
{
  return thrust::is_sorted_until(first, last, comp) == last;
} // end is_sorted()

template<typename ForwardIterator>
  ForwardIterator is_sorted_until(ForwardIterator first,
                                  ForwardIterator last)
{
  typedef typename thrust::iterator_value<ForwardIterator>::type InputType;

  return thrust::is_sorted_until(first, last, thrust::less<InputType>());
} // end is_sorted_until()

template<typename ForwardIterator,
         typename Compare>
  ForwardIterator is_sorted_until(ForwardIterator first,
                                  ForwardIterator last,
                                  Compare comp)
{
  if(thrust::distance(first,last) < 2) return last;

  typedef thrust::tuple<ForwardIterator,ForwardIterator> IteratorTuple;
  typedef thrust::zip_iterator<IteratorTuple>            ZipIterator;

  ForwardIterator first_plus_one = first;
  thrust::advance(first_plus_one, 1);

  ZipIterator zipped_first = thrust::make_zip_iterator(thrust::make_tuple(first_plus_one, first));
  ZipIterator zipped_last  = thrust::make_zip_iterator(thrust::make_tuple(last, first));

  return thrust::get<0>(thrust::find_if(zipped_first, zipped_last, thrust::detail::tuple_binary_predicate<Compare>(comp)).get_iterator_tuple());
} // end is_sorted_until()

} // end namespace thrust

