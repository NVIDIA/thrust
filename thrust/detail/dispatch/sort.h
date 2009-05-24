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

template<typename RandomAccessIterator>
  void sort(RandomAccessIterator begin,
            RandomAccessIterator end,
            thrust::random_access_host_iterator_tag)
{
    std::sort(begin, end);
}

template<typename RandomAccessIterator,
         typename StrictWeakCompare>
  void sort(RandomAccessIterator begin,
            RandomAccessIterator end,
            StrictWeakCompare comp,
            thrust::random_access_host_iterator_tag)
{
    std::sort(begin, end, comp);
}


template<typename RandomAccessIterator>
  void stable_sort(RandomAccessIterator begin,
                   RandomAccessIterator end,
                   thrust::random_access_host_iterator_tag)
{
    std::stable_sort(begin,end);
}


template<typename RandomAccessIterator,
         typename StrictWeakCompare>
  void stable_sort(RandomAccessIterator begin,
                   RandomAccessIterator end,
                   StrictWeakCompare comp,
                   thrust::random_access_host_iterator_tag)
{
    std::stable_sort(begin,end,comp);
}

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator>
  void stable_sort_by_key(RandomAccessKeyIterator keys_begin,
                          RandomAccessKeyIterator keys_end,
                          RandomAccessValueIterator values_begin,
                          thrust::random_access_host_iterator_tag,
                          thrust::random_access_host_iterator_tag)
{
    thrust::sorting::stable_merge_sort_by_key(keys_begin, keys_end, values_begin);
}

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator,
         typename StrictWeakOrdering>
  void stable_sort_by_key(RandomAccessKeyIterator keys_begin,
                          RandomAccessKeyIterator keys_end,
                          RandomAccessValueIterator values_begin,
                          StrictWeakOrdering comp,
                          thrust::random_access_host_iterator_tag,
                          thrust::random_access_host_iterator_tag)

{
    thrust::sorting::stable_merge_sort_by_key(keys_begin, keys_end, values_begin, comp);
}


////////////////////
/// DEVICE PATHS ///
////////////////////

template<typename RandomAccessIterator>
  void sort(RandomAccessIterator begin,
            RandomAccessIterator end,
            thrust::random_access_device_iterator_tag)
{
    // pass to thrust::stable_sort
    thrust::stable_sort(begin, end);
}

// TODO add device in function name
template<typename RandomAccessIterator>
  void stable_sort_pod(RandomAccessIterator begin,
                       RandomAccessIterator end,
                       thrust::detail::true_type)
{
    // device path for thrust::stable_sort with plain old data (POD) keys,
    // (e.g. int, float, short, etc.) is implemented with stable_radix_sort
    thrust::sorting::stable_radix_sort(begin, end);
}

template<typename RandomAccessIterator>
  void stable_sort_pod(RandomAccessIterator begin,
                       RandomAccessIterator end,
                       thrust::detail::false_type)
{
    // device path for thrust::stable_sort with non-POD keys is implemented
    // with thrust::stable_merge_sort
    thrust::sorting::stable_merge_sort(begin, end);
}

template<typename RandomAccessIterator>
  void stable_sort(RandomAccessIterator begin,
                   RandomAccessIterator end,
                   thrust::random_access_device_iterator_tag)
{
    // dispatch on whether KeyType is PlainOldData
    typedef typename thrust::iterator_traits<RandomAccessIterator>::value_type KeyType;
    stable_sort_pod(begin, end, thrust::detail::is_pod<KeyType>());
}

template<typename RandomAccessIterator,
         typename StrictWeakCompare>
  void sort(RandomAccessIterator begin,
            RandomAccessIterator end,
            StrictWeakCompare comp,
            thrust::random_access_device_iterator_tag)
{
    // just pass to thrust::stable_sort
    thrust::stable_sort(begin, end, comp);
}

template<typename RandomAccessIterator,
         typename StrictWeakCompare>
  void stable_sort(RandomAccessIterator begin,
                   RandomAccessIterator end,
                   StrictWeakCompare comp,
                   thrust::random_access_device_iterator_tag)
{
    // use stable_merge_sort for general comparison methods
    thrust::sorting::stable_merge_sort(begin, end, comp);
}

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator>
  void stable_sort_by_key_device_pod_switch(RandomAccessKeyIterator keys_begin,
                                            RandomAccessKeyIterator keys_end,
                                            RandomAccessValueIterator values_begin,
                                            thrust::detail::true_type)
{
    // device path for thrust::stable_sort_by_key with plain old data (POD) keys,
    // (e.g. int, float, short, etc.) is implemented with stable_radix_sort_by_key
    thrust::sorting::stable_radix_sort_by_key(keys_begin, keys_end, values_begin);
}

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator>
  void stable_sort_by_key_device_pod_switch(RandomAccessKeyIterator keys_begin,
                                            RandomAccessKeyIterator keys_end,
                                            RandomAccessValueIterator values_begin,
                                            thrust::detail::false_type)
{
    // device path for thrust::stable_sort_by_key with non-POD keys is implemented
    // with thrust::stable_merge_sort_by_key
    thrust::sorting::stable_merge_sort_by_key(keys_begin, keys_end, values_begin);
}

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator>
  void stable_sort_by_key(RandomAccessKeyIterator keys_begin,
                          RandomAccessKeyIterator keys_end,
                          RandomAccessValueIterator values_begin,
                          thrust::random_access_device_iterator_tag,
                          thrust::random_access_device_iterator_tag)
{
    typedef typename thrust::iterator_traits<RandomAccessKeyIterator>::value_type KeyType;
    stable_sort_by_key_device_pod_switch
        (keys_begin, keys_end, values_begin, thrust::detail::is_pod<KeyType>());
}

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator,
         typename StrictWeakOrdering>
  void stable_sort_by_key(RandomAccessKeyIterator keys_begin,
                          RandomAccessKeyIterator keys_end,
                          RandomAccessValueIterator values_begin,
                          StrictWeakOrdering comp,
                          thrust::random_access_device_iterator_tag,
                          thrust::random_access_device_iterator_tag)
{
    thrust::sorting::stable_merge_sort_by_key(keys_begin, keys_end, values_begin, comp);
}

} // end namespace dispatch

} // end namespace detail

} // end namespace thrust

