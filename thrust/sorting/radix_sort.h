/*
 *  Copyright 2008-2010 NVIDIA Corporation
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
#include <thrust/sort.h>

#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
#pragma message("-----------------------------------------------------------------------")
#pragma message("| DEPRECATION WARNING:                                                 ")
#pragma message("| thrust/sorting/radix_sort.h has been deprecated and will be removed  ")
#pragma message("| Use the sorting functions defined in thrust/sort.h instead           ")
#pragma message("-----------------------------------------------------------------------")
#else
#warning -----------------------------------------------------------------------
#warning | DEPRECATION WARNING: 
#warning | thrust/sorting/radix_sort.h has been deprecated and will be removed
#warning | Use the sorting functions defined in thrust/sort.h instead
#warning -----------------------------------------------------------------------
#endif // THRUST_HOST_COMPILER_MSVC

namespace thrust
{
namespace sorting
{

template<typename RandomAccessIterator>
  void radix_sort(RandomAccessIterator first,
                  RandomAccessIterator last)
{
    thrust::sort(first, last);
}

template<typename RandomAccessIterator>
  void stable_radix_sort(RandomAccessIterator first,
                         RandomAccessIterator last)
{
    thrust::stable_sort(first, last);
}

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void radix_sort_by_key(RandomAccessIterator1 keys_first,
                         RandomAccessIterator1 keys_last,
                         RandomAccessIterator2 values_first)
{
    thrust::sort_by_key(keys_first, keys_last, values_first);
}


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void stable_radix_sort_by_key(RandomAccessIterator1 keys_first,
                                RandomAccessIterator1 keys_last,
                                RandomAccessIterator2 values_first)
{
    thrust::stable_sort_by_key(keys_first, keys_last, values_first);
}

} // end namespace sorting
} // end namespace thrust

