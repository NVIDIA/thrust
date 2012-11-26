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

#include <thrust/detail/config.h>
#include <thrust/system/detail/generic/sort.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/find.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/detail/internal_functional.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename System,
         typename RandomAccessIterator>
  void sort(thrust::dispatchable<System> &system,
            RandomAccessIterator first,
            RandomAccessIterator last)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type; 
  thrust::sort(system, first, last, thrust::less<value_type>());
} // end sort()


template<typename System,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void sort(thrust::dispatchable<System> &system,
            RandomAccessIterator first,
            RandomAccessIterator last,
            StrictWeakOrdering comp)
{
  // implement with stable_sort
  thrust::stable_sort(system, first, last, comp);
} // end sort()


template<typename System,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void sort_by_key(thrust::dispatchable<System> &system,
                   RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type;
  thrust::sort_by_key(system, keys_first, keys_last, values_first, thrust::less<value_type>());
} // end sort_by_key()


template<typename System,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void sort_by_key(thrust::dispatchable<System> &system,
                   RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first,
                   StrictWeakOrdering comp)
{
  // implement with stable_sort_by_key
  thrust::stable_sort_by_key(system, keys_first, keys_last, values_first, comp);
} // end sort_by_key()


template<typename System,
         typename RandomAccessIterator>
  void stable_sort(thrust::dispatchable<System> &system,
                   RandomAccessIterator first,
                   RandomAccessIterator last)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;
  thrust::stable_sort(system, first, last, thrust::less<value_type>());
} // end stable_sort()


template<typename System,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void stable_sort_by_key(thrust::dispatchable<System> &system,
                          RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first)
{
  typedef typename iterator_value<RandomAccessIterator1>::type value_type;
  thrust::stable_sort_by_key(system, keys_first, keys_last, values_first, thrust::less<value_type>());
} // end stable_sort_by_key()


template<typename System, typename ForwardIterator>
  bool is_sorted(thrust::dispatchable<System> &system,
                 ForwardIterator first,
                 ForwardIterator last)
{
  return thrust::is_sorted_until(system, first, last) == last;
} // end is_sorted()


template<typename System,
         typename ForwardIterator,
         typename Compare>
  bool is_sorted(thrust::dispatchable<System> &system,
                 ForwardIterator first,
                 ForwardIterator last,
                 Compare comp)
{
  return thrust::is_sorted_until(system, first, last, comp) == last;
} // end is_sorted()


template<typename System, typename ForwardIterator>
  ForwardIterator is_sorted_until(thrust::dispatchable<System> &system,
                                  ForwardIterator first,
                                  ForwardIterator last)
{
  typedef typename thrust::iterator_value<ForwardIterator>::type InputType;

  return thrust::is_sorted_until(system, first, last, thrust::less<InputType>());
} // end is_sorted_until()


template<typename System,
         typename ForwardIterator,
         typename Compare>
  ForwardIterator is_sorted_until(thrust::dispatchable<System> &system,
                                  ForwardIterator first,
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

  return thrust::get<0>(thrust::find_if(system, zipped_first, zipped_last, thrust::detail::tuple_binary_predicate<Compare>(comp)).get_iterator_tuple());
} // end is_sorted_until()


template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_sort(tag,
                   RandomAccessIterator first,
                   RandomAccessIterator last,
                   StrictWeakOrdering comp)
{
  // unimplemented primitive
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<RandomAccessIterator, false>::value) );
} // end stable_sort()


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void stable_sort_by_key(tag,
                          RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first,
                          StrictWeakOrdering comp)
{
  // unimplemented primitive
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<RandomAccessIterator1, false>::value) );
} // end stable_sort_by_key()


} // end generic
} // end detail
} // end system
} // end thrust

