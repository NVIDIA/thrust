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


/*! \file merge_sort.h
 *  \brief Dispatches merge_sort based on iterator_category.
 */

#include <algorithm>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>

#include <thrust/sorting/detail/host/stable_merge_sort.h>
#include <thrust/sorting/detail/device/stable_merge_sort.h>


namespace thrust
{

namespace sorting
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
  void stable_merge_sort(RandomAccessIterator begin,
                         RandomAccessIterator end,
                         StrictWeakOrdering comp,
                         thrust::host_space_tag)
{
    thrust::sorting::detail::host::stable_merge_sort(begin, end, comp);
}

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void stable_merge_sort_by_key(RandomAccessIterator1 keys_begin,
                                RandomAccessIterator1 keys_end,
                                RandomAccessIterator2 values_begin,
                                StrictWeakOrdering comp,
                                thrust::host_space_tag,
                                thrust::host_space_tag)
{
    thrust::sorting::detail::host::stable_merge_sort_by_key(keys_begin, keys_end, values_begin, comp);
}


////////////////////
/// DEVICE PATHS ///
////////////////////

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_merge_sort(RandomAccessIterator begin,
                         RandomAccessIterator end,
                         StrictWeakOrdering comp,
                         thrust::device_space_tag)
{
    thrust::sorting::detail::device::stable_merge_sort(begin, end, comp);
}

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void stable_merge_sort_by_key(RandomAccessIterator1 keys_begin,
                                RandomAccessIterator1 keys_end,
                                RandomAccessIterator2 values_begin,
                                StrictWeakOrdering comp,
                                thrust::device_space_tag,
                                thrust::device_space_tag)
{
    thrust::sorting::detail::device::stable_merge_sort_by_key(keys_begin, keys_end, values_begin, comp);
} 

} // end namespace dispatch

} // end namespace detail

} // end namespace sorting

} // end namespace thrust

