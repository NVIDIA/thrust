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


/*! \file gather.inl
 *  \brief Inline file for gather.h
 */

#pragma once

#include <thrust/distance.h>

#include <thrust/detail/device/for_each.h>
#include <thrust/detail/device/dereference.h>

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

template <typename RandomAccessIterator>
struct gather_functor
{
  RandomAccessIterator input;

  gather_functor(RandomAccessIterator _input)
    : input(_input) {}
  
  template <typename Tuple>
  __device__
  void operator()(Tuple t)
  { 
    thrust::get<0>(t) = thrust::detail::device::dereference(input, thrust::get<1>(t)); 
  }
}; // end gather_functor

template <typename RandomAccessIterator, typename Predicate>
struct gather_if_functor
{
  RandomAccessIterator input;
  Predicate pred;

  gather_if_functor(RandomAccessIterator _input, Predicate _pred)
    : input(_input), pred(_pred) {}
  
  template <typename Tuple>
  __device__
  void operator()(Tuple t)
  { 
    if(pred(thrust::get<2>(t)))
      thrust::get<0>(t) = thrust::detail::device::dereference(input, thrust::get<1>(t)); 
  }
}; // end gather_functor

} // end detail


template<typename ForwardIterator,
         typename InputIterator,
         typename RandomAccessIterator>
  void gather(ForwardIterator first,
              ForwardIterator last,
              InputIterator  map,
              RandomAccessIterator input)
{
  detail::gather_functor<RandomAccessIterator> func(input);
  thrust::detail::device::for_each(thrust::make_zip_iterator(thrust::make_tuple(first, map)),
                                   thrust::make_zip_iterator(thrust::make_tuple(last,  map + thrust::distance(first, last))),
                                   func);
} // end gather()


template<typename ForwardIterator,
         typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename Predicate>
  void gather_if(ForwardIterator first,
                 ForwardIterator last,
                 InputIterator1 map,
                 InputIterator2 stencil,
                 RandomAccessIterator input,
                 Predicate pred)
{
  detail::gather_if_functor<RandomAccessIterator, Predicate> func(input, pred);
  thrust::detail::device::for_each(thrust::make_zip_iterator(thrust::make_tuple(first, map, stencil)),
                                   thrust::make_zip_iterator(thrust::make_tuple(last,  map + thrust::distance(first, last), stencil + thrust::distance(first, last))),
                                   func);
} // end gather_if()


} // end namespace generic
} // end namespace device
} // end namespace detail
} // end namespace thrust

