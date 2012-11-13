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
#include <thrust/pair.h>
#include <thrust/system/cpp/detail/tag.h>
#include <thrust/system/detail/internal/scalar/merge.h>

namespace thrust
{
namespace system
{
namespace cpp
{
namespace detail
{

template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
OutputIterator merge(dispatchable<System> &,
                     InputIterator1 first1,
                     InputIterator1 last1,
                     InputIterator2 first2,
                     InputIterator2 last2,
                     OutputIterator result,
                     StrictWeakOrdering comp)
{
  return thrust::system::detail::internal::scalar::merge(first1, last1, first2, last2, result, comp);
}

template <typename System,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakOrdering>
thrust::pair<OutputIterator1,OutputIterator2>
  merge_by_key(dispatchable<System> &,
               InputIterator1 keys_first1,
               InputIterator1 keys_last1,
               InputIterator2 keys_first2,
               InputIterator2 keys_last2,
               InputIterator3 values_first1,
               InputIterator4 values_first2,
               OutputIterator1 keys_result,
               OutputIterator2 values_result,
               StrictWeakOrdering comp)
{
  return thrust::system::detail::internal::scalar::merge_by_key(keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result, comp);
}

} // end namespace detail
} // end namespace cpp
} // end namespace system
} // end namespace thrust

