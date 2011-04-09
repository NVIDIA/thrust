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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/detail/minimum_space.h>
#include <thrust/detail/copy.h>
#include <algorithm>

namespace thrust
{
namespace detail
{
namespace dispatch
{

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  RandomAccessIterator2 overlapped_copy(RandomAccessIterator1 first,
                                        RandomAccessIterator1 last,
                                        RandomAccessIterator2 result,
                                        thrust::host_space_tag)
{
  if(first < last && first <= result && result < last)
  {
    // result lies in [first, last)
    // it's safe to use std::copy_backward here
    result = std::copy_backward(first, last, result + (last - first));
  } // end if
  else
  {
    // result + (last - first) lies in [first, last)
    // it's safe to use std::copy here
    result = std::copy(first, last, result);
  } // end else

  return result;
} // end overlapped_copy()


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  RandomAccessIterator2 overlapped_copy(RandomAccessIterator1 first,
                                        RandomAccessIterator1 last,
                                        RandomAccessIterator2 result,
                                        thrust::device_space_tag)
{
  typedef typename thrust::iterator_space<RandomAccessIterator1>::type space1;
  typedef typename thrust::iterator_space<RandomAccessIterator2>::type space2;

  typedef typename thrust::detail::minimum_space<space1,space2>::type space;
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type;

  // make a temporary copy of [first,last), and copy into it first
  thrust::detail::raw_buffer<value_type, space> temp(first,last);
  return thrust::copy(temp.begin(), temp.end(), result);
} // end overlapped_copy()

} // end dispatch


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  RandomAccessIterator2 overlapped_copy(RandomAccessIterator1 first,
                                        RandomAccessIterator1 last,
                                        RandomAccessIterator2 result)
{
  return thrust::detail::dispatch::overlapped_copy(first, last, result,
      typename thrust::detail::minimum_space<
        typename thrust::iterator_space<RandomAccessIterator1>::type,
        typename thrust::iterator_space<RandomAccessIterator2>::type
      >::type());
} // end overlapped_copy()

} // end detail
} // end thrust

