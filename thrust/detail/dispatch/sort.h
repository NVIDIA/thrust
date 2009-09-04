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


/*! \file sort.h
 *  \brief Defines the interface to the
 *         family of sort functions.
 */

#include <algorithm>

#include <thrust/sort.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>

#include <thrust/sorting/radix_sort.h>
#include <thrust/sorting/merge_sort.h>

namespace thrust
{

namespace detail
{

namespace dispatch
{

//////////////////
/// HOST PATHS ///
//////////////////

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void sort(RandomAccessIterator begin,
            RandomAccessIterator end,
            StrictWeakOrdering comp,
            thrust::host_space_tag)
{
    // std::sort(begin,end,comp);  // doesn't support zip_iterator
    thrust::sorting::merge_sort(begin, end, comp);
}


template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_sort(RandomAccessIterator begin,
                   RandomAccessIterator end,
                   StrictWeakOrdering comp,
                   thrust::host_space_tag)
{
    // std::stable_sort(begin,end,comp);  // doesn't support zip_iterator
    thrust::sorting::stable_merge_sort(begin, end, comp);
}

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator,
         typename StrictWeakOrdering>
  void stable_sort_by_key(RandomAccessKeyIterator keys_begin,
                          RandomAccessKeyIterator keys_end,
                          RandomAccessValueIterator values_begin,
                          StrictWeakOrdering comp,
                          thrust::host_space_tag,
                          thrust::host_space_tag)

{
    thrust::sorting::stable_merge_sort_by_key(keys_begin, keys_end, values_begin, comp);
}


////////////////////
/// DEVICE PATHS ///
////////////////////

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_sort_with_radix_sort(RandomAccessIterator begin,
                                   RandomAccessIterator end,
                                   StrictWeakOrdering comp,
                                   thrust::detail::true_type)
{
    // device path for thrust::stable_sort with primitive keys
    // (e.g. int, float, short, etc.) and the default less<T> comparison
    // method is implemented with stable_radix_sort
    thrust::sorting::stable_radix_sort(begin, end);
}

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_sort_with_radix_sort(RandomAccessIterator begin,
                                   RandomAccessIterator end,
                                   StrictWeakOrdering comp,
                                   thrust::detail::false_type)
{
    // device path for thrust::stable_sort with general keys 
    // and comparison methods is implemented with stable_merge_sort
    thrust::sorting::stable_merge_sort(begin, end, comp);
}

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator,
         typename StrictWeakOrdering>
  void stable_sort_by_key_with_radix_sort(RandomAccessKeyIterator keys_begin,
                                          RandomAccessKeyIterator keys_end,
                                          RandomAccessValueIterator values_begin,
                                          StrictWeakOrdering comp,
                                          thrust::detail::true_type)
{
    // device path for thrust::stable_sort_by_key with primitive keys
    // (e.g. int, float, short, etc.) and the default less<T> comparison
    // method is implemented with stable_radix_sort_by_key
    thrust::sorting::stable_radix_sort_by_key(keys_begin, keys_end, values_begin);
}

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator,
         typename StrictWeakOrdering>
  void stable_sort_by_key_with_radix_sort(RandomAccessKeyIterator keys_begin,
                                          RandomAccessKeyIterator keys_end,
                                          RandomAccessValueIterator values_begin,
                                          StrictWeakOrdering comp,
                                          thrust::detail::false_type)
{
    // device path for thrust::stable_sort with general keys 
    // and comparison methods is implemented with stable_merge_sort
    thrust::sorting::stable_merge_sort_by_key(keys_begin, keys_end, values_begin, comp);
}


// XXX entry points
template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void sort(RandomAccessIterator begin,
            RandomAccessIterator end,
            StrictWeakOrdering comp,
            thrust::device_space_tag)
{
    // XXX forward to thrust::stable_sort
    thrust::stable_sort(begin, end, comp);
}

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_sort(RandomAccessIterator begin,
                   RandomAccessIterator end,
                   StrictWeakOrdering comp,
                   thrust::device_space_tag)
{
    // dispatch on whether we can use radix_sort
    typedef typename thrust::iterator_traits<RandomAccessIterator>::value_type KeyType;
    static const bool use_radix_sort = thrust::detail::is_pod<KeyType>::value &&
                                       thrust::detail::is_same<StrictWeakOrdering, typename thrust::less<KeyType> >::value;

    stable_sort_with_radix_sort(begin, end, comp, thrust::detail::integral_constant<bool, use_radix_sort>());
}


template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator,
         typename StrictWeakOrdering>
  void stable_sort_by_key(RandomAccessKeyIterator keys_begin,
                          RandomAccessKeyIterator keys_end,
                          RandomAccessValueIterator values_begin,
                          StrictWeakOrdering comp,
                          thrust::device_space_tag,
                          thrust::device_space_tag)
{
    // dispatch on whether we can use radix_sort
    typedef typename thrust::iterator_traits<RandomAccessKeyIterator>::value_type KeyType;
    static const bool use_radix_sort = thrust::detail::is_pod<KeyType>::value &&
                                       thrust::detail::is_same<StrictWeakOrdering, typename thrust::less<KeyType> >::value;

    stable_sort_by_key_with_radix_sort(keys_begin, keys_end, values_begin, comp,
            thrust::detail::integral_constant<bool, use_radix_sort>());
}

} // end namespace dispatch

} // end namespace detail

} // end namespace thrust

