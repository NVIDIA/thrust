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

#include <thrust/detail/config.h>
#include <thrust/sort.h>
#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/backend/generic/select_system.h>
#include <thrust/detail/backend/generic/sort.h>

// XXX make the backend-specific versions available
// XXX try to eliminate the need for these
#include <thrust/detail/backend/cpp/sort.h>
#include <thrust/detail/backend/omp/sort.h>
#include <thrust/detail/backend/cuda/sort.h>

#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/find.h>

namespace thrust
{

///////////////
// Key Sorts //
///////////////

template<typename RandomAccessIterator>
  void sort(RandomAccessIterator first,
            RandomAccessIterator last)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::sort;

  typedef typename thrust::iterator_space<RandomAccessIterator>::type space;

  return sort(select_system(space()), first, last);
} // end sort()


template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void sort(RandomAccessIterator first,
            RandomAccessIterator last,
            StrictWeakOrdering comp)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::sort;

  typedef typename thrust::iterator_space<RandomAccessIterator>::type space;

  return sort(select_system(space()), first, last, comp);
} // end sort()


template<typename RandomAccessIterator>
  void stable_sort(RandomAccessIterator first,
                   RandomAccessIterator last)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::stable_sort;

  typedef typename thrust::iterator_space<RandomAccessIterator>::type space;

  return stable_sort(select_system(space()), first, last);
} // end stable_sort() 


template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_sort(RandomAccessIterator first,
                   RandomAccessIterator last,
                   StrictWeakOrdering comp)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::stable_sort;

  typedef typename thrust::iterator_space<RandomAccessIterator>::type space;

  return stable_sort(select_system(space()), first, last, comp);
} // end stable_sort()



/////////////////////
// Key-Value Sorts //
/////////////////////

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void sort_by_key(RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::sort_by_key;

  typedef typename thrust::iterator_space<RandomAccessIterator1>::type space1;
  typedef typename thrust::iterator_space<RandomAccessIterator2>::type space2;

  return sort_by_key(select_system(space1(),space2()), keys_first, keys_last, values_first);
} // end sort_by_key()


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void sort_by_key(RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first,
                   StrictWeakOrdering comp)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::sort_by_key;

  typedef typename thrust::iterator_space<RandomAccessIterator1>::type space1;
  typedef typename thrust::iterator_space<RandomAccessIterator2>::type space2;

  return sort_by_key(select_system(space1(),space2()), keys_first, keys_last, values_first, comp);
} // end sort_by_key()


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void stable_sort_by_key(RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::stable_sort_by_key;

  typedef typename thrust::iterator_space<RandomAccessIterator1>::type space1;
  typedef typename thrust::iterator_space<RandomAccessIterator2>::type space2;

  return stable_sort_by_key(select_system(space1(),space2()), keys_first, keys_last, values_first);
} // end stable_sort_by_key()


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void stable_sort_by_key(RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first,
                          StrictWeakOrdering comp)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::stable_sort_by_key;

  typedef typename thrust::iterator_space<RandomAccessIterator1>::type space1;
  typedef typename thrust::iterator_space<RandomAccessIterator2>::type space2;

  return stable_sort_by_key(select_system(space1(),space2()), keys_first, keys_last, values_first, comp);
} // end stable_sort_by_key()


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

