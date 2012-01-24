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

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
OutputIterator merge(tag,
                     InputIterator1 first1,
                     InputIterator1 last1,
                     InputIterator2 first2,
                     InputIterator2 last2,
                     OutputIterator result,
                     StrictWeakOrdering comp)
{
  return thrust::system::detail::internal::scalar::merge(first1, last1, first2, last2, result, comp);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakOrdering>
thrust::pair<OutputIterator1,OutputIterator2>
    merge_by_key(tag,
                 InputIterator1 first1,
                 InputIterator1 last1,
                 InputIterator2 first2,
                 InputIterator2 last2,
                 InputIterator3 first3,
                 InputIterator4 first4,
                 OutputIterator1 output1,
                 OutputIterator2 output2,
                 StrictWeakOrdering comp)
{
  return thrust::system::detail::internal::scalar::merge_by_key(first1, last1, first2, last2, first3, first4, output1, output2, comp);
}

} // end namespace detail
} // end namespace cpp
} // end namespace system
} // end namespace thrust

