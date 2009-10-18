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


/*! \file scatter.inl
 *  \brief Inline file for scatter.h
 */

#pragma once

#include <thrust/detail/device/dereference.h>
#include <thrust/detail/device/for_each.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/distance.h>

namespace thrust
{

namespace detail
{

namespace device
{

// XXX WAR circluar #inclusion with this forward declaration
template<typename InputIterator, typename UnaryFunction> void for_each(InputIterator, InputIterator, UnaryFunction);

namespace detail
{


template <typename RandomAccessIterator>
struct scatter_functor
{
  RandomAccessIterator output;

  scatter_functor(RandomAccessIterator _output) 
    : output(_output) {}
  
  template <typename Tuple>
  __device__
  void operator()(Tuple t)
  { 
    thrust::detail::device::dereference(output, thrust::get<1>(t)) = thrust::get<0>(t); 
  }
}; // end scatter_functor


template <typename RandomAccessIterator, typename Predicate>
struct scatter_if_functor
{
  RandomAccessIterator output;
  Predicate pred;

  scatter_if_functor(RandomAccessIterator _output, Predicate _pred)
    : output(_output), pred(_pred) {}
  
  template <typename Tuple>
  __device__
  void operator()(Tuple t)
  { 
    if(pred(thrust::get<2>(t)))
      thrust::detail::device::dereference(output, thrust::get<1>(t)) = thrust::get<0>(t); 
  }
}; // end scatter_if_functor

} // end detail


template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
  void scatter(InputIterator1 first,
               InputIterator1 last,
               InputIterator2 map,
               RandomAccessIterator output)
{
  detail::scatter_functor<RandomAccessIterator> func(output);
  thrust::detail::device::for_each(thrust::make_zip_iterator(make_tuple(first, map)),
                                   thrust::make_zip_iterator(make_tuple(last,  map + thrust::distance(first, last))),
                                   func);
} // end scatter()


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator,
         typename Predicate>
  void scatter_if(InputIterator1 first,
                  InputIterator1 last,
                  InputIterator2 map,
                  InputIterator3 stencil,
                  RandomAccessIterator output,
                  Predicate pred)
{
  detail::scatter_if_functor<RandomAccessIterator, Predicate> func(output, pred);
  thrust::detail::device::for_each(thrust::make_zip_iterator(make_tuple(first, map, stencil)),
                                   thrust::make_zip_iterator(make_tuple(last,  map + thrust::distance(first, last), stencil + thrust::distance(first, last))),
                                   func);
} // end scatter_if()


} // end namespace device

} // end namespace detail

} // end namespace thrust

