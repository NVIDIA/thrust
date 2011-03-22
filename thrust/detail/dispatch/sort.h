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


/*! \file sort.h
 *  \brief Dispatch layer for sort functions.
 */

#include <thrust/iterator/iterator_categories.h>

#include <thrust/detail/host/sort.h>
#include <thrust/detail/device/sort.h>

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
    thrust::detail::host::sort(begin, end, comp);
}

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_sort(RandomAccessIterator begin,
                   RandomAccessIterator end,
                   StrictWeakOrdering comp,
                   thrust::host_space_tag)
{
    thrust::detail::host::stable_sort(begin, end, comp);
}

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator,
         typename StrictWeakOrdering>
  void sort_by_key(RandomAccessKeyIterator keys_begin,
                   RandomAccessKeyIterator keys_end,
                   RandomAccessValueIterator values_begin,
                   StrictWeakOrdering comp,
                   thrust::host_space_tag,
                   thrust::host_space_tag)

{
    thrust::detail::host::sort_by_key(keys_begin, keys_end, values_begin, comp);
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
    thrust::detail::host::stable_sort_by_key(keys_begin, keys_end, values_begin, comp);
}


////////////////////
/// DEVICE PATHS ///
////////////////////

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void sort(RandomAccessIterator begin,
            RandomAccessIterator end,
            StrictWeakOrdering comp,
            thrust::device_space_tag)
{
    thrust::detail::device::sort(begin, end, comp);
}

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_sort(RandomAccessIterator begin,
                   RandomAccessIterator end,
                   StrictWeakOrdering comp,
                   thrust::device_space_tag)
{
    thrust::detail::device::stable_sort(begin, end, comp);
}

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator,
         typename StrictWeakOrdering>
  void sort_by_key(RandomAccessKeyIterator keys_begin,
                   RandomAccessKeyIterator keys_end,
                   RandomAccessValueIterator values_begin,
                   StrictWeakOrdering comp,
                   thrust::device_space_tag,
                   thrust::device_space_tag)
{
    thrust::detail::device::sort_by_key(keys_begin, keys_end, values_begin, comp);
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
    thrust::detail::device::stable_sort_by_key(keys_begin, keys_end, values_begin, comp);
}

} // end namespace dispatch
} // end namespace detail
} // end namespace thrust

