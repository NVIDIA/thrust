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


/*! \file swap_ranges.h
 *  \brief Device implementation for swap_ranges.
 */

#pragma once

#include <thrust/distance.h>
#include <thrust/tuple.h>
#include <thrust/utility.h>

#include <thrust/iterator/zip_iterator.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace generic
{
namespace detail
{

struct swap_pair_elements
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  { 
    thrust::swap(thrust::get<0>(t), thrust::get<1>(t));
  }
}; // end swap_pair_elements

} // end namespace detail


template<typename ForwardIterator1,
         typename ForwardIterator2>
  ForwardIterator2 swap_ranges(ForwardIterator1 first1,
                               ForwardIterator1 last1,
                               ForwardIterator2 first2)
{
  thrust::detail::device::for_each(thrust::make_zip_iterator(thrust::make_tuple(first1, first2)),
                                   thrust::make_zip_iterator(thrust::make_tuple(last1,  first2 + thrust::distance(first1, last1))),
                                   detail::swap_pair_elements());

  return first2 + thrust::distance(first1, last1);
}

} // end namespace generic
} // end namespace device
} // end namespace detail
} // end namespace thrust

