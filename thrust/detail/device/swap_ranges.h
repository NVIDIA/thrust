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


/*! \file swap_ranges.h
 *  \brief Device implementation for swap_ranges.
 */

#pragma once

#include <thrust/distance.h>
#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/device/cuda/vectorize.h>
#include <thrust/detail/device/dereference.h>
#include <thrust/utility.h>

namespace thrust
{

namespace detail
{

namespace device
{

namespace detail
{

template <typename ForwardIterator1, typename ForwardIterator2>
struct swap_ranges_functor
{
    ForwardIterator1 first1;
    ForwardIterator2 first2;

    swap_ranges_functor(ForwardIterator1 _first1, ForwardIterator2 _first2)
        : first1(_first1), first2(_first2) {}

    template <typename IntegerType>
        __device__
        void operator()(const IntegerType i)
        { 
            thrust::swap(thrust::detail::device::dereference(first1, i), thrust::detail::device::dereference(first2, i));
        }
}; // end swap_ranges_functor

} // end namespace detail


template<typename ForwardIterator1,
         typename ForwardIterator2>
  ForwardIterator2 swap_ranges(ForwardIterator1 first1,
                               ForwardIterator1 last1,
                               ForwardIterator2 first2)
{
    detail::swap_ranges_functor<ForwardIterator1,ForwardIterator2> func(first1, first2);

    thrust::detail::device::cuda::vectorize(last1 - first1, func);

    return first2 + thrust::distance(first1, last1);
}

} // end namespace device

} // end namespace detail

} // end namespace thrust

