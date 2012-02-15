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
#include <thrust/detail/type_traits.h>
#include <thrust/detail/backend/cpp/detail/general_copy.h>
#include <thrust/detail/backend/cpp/detail/trivial_copy.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace cpp
{
namespace dispatch
{


template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result,
                      thrust::detail::false_type)
{
  return thrust::detail::backend::cpp::detail::general_copy(first, last, result);
} // end copy()


template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result,
                        thrust::detail::false_type)
{
  return thrust::detail::backend::cpp::detail::general_copy_n(first, n, result);
} // end copy_n()


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  RandomAccessIterator2 copy(RandomAccessIterator1 first,
                             RandomAccessIterator1 last,
                             RandomAccessIterator2 result,
                             thrust::detail::true_type)
{
  typedef typename thrust::iterator_difference<RandomAccessIterator1>::type Size;

  const Size n = last - first;
  thrust::detail::backend::cpp::detail::trivial_copy_n(&*first, n, &*result);
  return result + n;
} // end copy()


template<typename RandomAccessIterator1,
         typename Size,
         typename RandomAccessIterator2>
  RandomAccessIterator2 copy_n(RandomAccessIterator1 first,
                               Size n,
                               RandomAccessIterator2 result,
                               thrust::detail::true_type)
{
  thrust::detail::backend::cpp::detail::trivial_copy_n(&*first, n, &*result);
  return result + n;
} // end copy_n()


} // end dispatch
} // end cpp
} // end backend
} // end detail
} // end thrust

