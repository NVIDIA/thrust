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

#include <komrade/sort.h>
#include <komrade/iterator/iterator_categories.h>
#include <komrade/iterator/iterator_traits.h>
#include <komrade/detail/type_traits.h>

#include <komrade/sorting/radix_sort.h>
#include <komrade/sorting/merge_sort.h>

namespace komrade
{

namespace detail
{

namespace dispatch
{

//////////////////
/// HOST PATHS ///
//////////////////

template<typename RandomAccessIterator>
  void sort(RandomAccessIterator begin,
            RandomAccessIterator end,
            komrade::random_access_host_iterator_tag)
{
    std::sort(begin, end);
}

template<typename RandomAccessIterator,
         typename StrictWeakCompare>
  void sort(RandomAccessIterator begin,
            RandomAccessIterator end,
            StrictWeakCompare comp,
            komrade::random_access_host_iterator_tag)
{
    std::sort(begin, end, comp);
}


template<typename RandomAccessIterator>
  void stable_sort(RandomAccessIterator begin,
                   RandomAccessIterator end,
                   komrade::random_access_host_iterator_tag)
{
    std::stable_sort(begin,end);
}


template<typename RandomAccessIterator,
         typename StrictWeakCompare>
  void stable_sort(RandomAccessIterator begin,
                   RandomAccessIterator end,
                   StrictWeakCompare comp,
                   komrade::random_access_host_iterator_tag)
{
    std::stable_sort(begin,end,comp);
}

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator>
  void stable_sort_by_key(RandomAccessKeyIterator keys_begin,
                          RandomAccessKeyIterator keys_end,
                          RandomAccessValueIterator values_begin,
                          komrade::random_access_host_iterator_tag,
                          komrade::random_access_host_iterator_tag)
{
    komrade::sorting::stable_merge_sort_by_key(keys_begin, keys_end, values_begin);
}

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator,
         typename StrictWeakOrdering>
  void stable_sort_by_key(RandomAccessKeyIterator keys_begin,
                          RandomAccessKeyIterator keys_end,
                          RandomAccessValueIterator values_begin,
                          StrictWeakOrdering comp,
                          komrade::random_access_host_iterator_tag,
                          komrade::random_access_host_iterator_tag)

{
    komrade::sorting::stable_merge_sort_by_key(keys_begin, keys_end, values_begin, comp);
}


////////////////////
/// DEVICE PATHS ///
////////////////////

template<typename RandomAccessIterator>
  void sort(RandomAccessIterator begin,
            RandomAccessIterator end,
            komrade::random_access_device_iterator_tag)
{
    // pass to komrade::stable_sort
    komrade::stable_sort(begin, end);
}

// TODO add device in function name
template<typename RandomAccessIterator>
  void stable_sort_pod(RandomAccessIterator begin,
                       RandomAccessIterator end,
                       komrade::detail::true_type)
{
    // device path for komrade::stable_sort with plain old data (POD) keys,
    // (e.g. int, float, short, etc.) is implemented with stable_radix_sort
    komrade::sorting::stable_radix_sort(begin, end);
}

template<typename RandomAccessIterator>
  void stable_sort_pod(RandomAccessIterator begin,
                       RandomAccessIterator end,
                       komrade::detail::false_type)
{
    // device path for komrade::stable_sort with non-POD keys is implemented
    // with komrade::stable_merge_sort
    komrade::sorting::stable_merge_sort(begin, end);
}

template<typename RandomAccessIterator>
  void stable_sort(RandomAccessIterator begin,
                   RandomAccessIterator end,
                   komrade::random_access_device_iterator_tag)
{
    // dispatch on whether KeyType is PlainOldData
    typedef typename komrade::iterator_traits<RandomAccessIterator>::value_type KeyType;
    stable_sort_pod(begin, end, komrade::detail::is_pod<KeyType>());
}

template<typename RandomAccessIterator,
         typename StrictWeakCompare>
  void sort(RandomAccessIterator begin,
            RandomAccessIterator end,
            StrictWeakCompare comp,
            komrade::random_access_device_iterator_tag)
{
    // just pass to komrade::stable_sort
    komrade::stable_sort(begin, end, comp);
}

template<typename RandomAccessIterator,
         typename StrictWeakCompare>
  void stable_sort(RandomAccessIterator begin,
                   RandomAccessIterator end,
                   StrictWeakCompare comp,
                   komrade::random_access_device_iterator_tag)
{
    // use stable_merge_sort for general comparison methods
    komrade::sorting::stable_merge_sort(begin, end, comp);
}

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator>
  void stable_sort_by_key_device_pod_switch(RandomAccessKeyIterator keys_begin,
                                            RandomAccessKeyIterator keys_end,
                                            RandomAccessValueIterator values_begin,
                                            komrade::detail::true_type)
{
    // device path for komrade::stable_sort_by_key with plain old data (POD) keys,
    // (e.g. int, float, short, etc.) is implemented with stable_radix_sort_by_key
    komrade::sorting::stable_radix_sort_by_key(keys_begin, keys_end, values_begin);
}

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator>
  void stable_sort_by_key_device_pod_switch(RandomAccessKeyIterator keys_begin,
                                            RandomAccessKeyIterator keys_end,
                                            RandomAccessValueIterator values_begin,
                                            komrade::detail::false_type)
{
    // device path for komrade::stable_sort_by_key with non-POD keys is implemented
    // with komrade::stable_merge_sort_by_key
    komrade::sorting::stable_merge_sort_by_key(keys_begin, keys_end, values_begin);
}

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator>
  void stable_sort_by_key(RandomAccessKeyIterator keys_begin,
                          RandomAccessKeyIterator keys_end,
                          RandomAccessValueIterator values_begin,
                          komrade::random_access_device_iterator_tag,
                          komrade::random_access_device_iterator_tag)
{
    typedef typename komrade::iterator_traits<RandomAccessKeyIterator>::value_type KeyType;
    stable_sort_by_key_device_pod_switch
        (keys_begin, keys_end, values_begin, komrade::detail::is_pod<KeyType>());
}

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator,
         typename StrictWeakOrdering>
  void stable_sort_by_key(RandomAccessKeyIterator keys_begin,
                          RandomAccessKeyIterator keys_end,
                          RandomAccessValueIterator values_begin,
                          StrictWeakOrdering comp,
                          komrade::random_access_device_iterator_tag,
                          komrade::random_access_device_iterator_tag)
{
    komrade::sorting::stable_merge_sort_by_key(keys_begin, keys_end, values_begin, comp);
}

} // end namespace dispatch

} // end namespace detail

} // end namespace komrade

