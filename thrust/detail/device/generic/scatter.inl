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


/*! \file scatter.inl
 *  \brief Inline file for scatter.h
 */

#pragma once

#include <thrust/detail/device/dereference.h>
#include <thrust/detail/device/for_each.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/distance.h>
#include <thrust/iterator/detail/minimum_space.h>
#include <thrust/iterator/detail/forced_iterator.h>

namespace thrust
{
namespace detail
{
namespace device
{

// XXX WAR circluar #inclusion with this forward declaration
template<typename InputIterator, typename UnaryFunction> void for_each(InputIterator, InputIterator, UnaryFunction);

namespace generic
{
namespace detail
{

template <typename RandomAccessIterator>
struct scatter_functor
{
  RandomAccessIterator output;

  scatter_functor(RandomAccessIterator _output) 
    : output(_output) {}
  
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  { 
    RandomAccessIterator dst = output + thrust::get<1>(t);
    thrust::detail::device::dereference(dst) = thrust::get<0>(t);
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
  __host__ __device__
  void operator()(Tuple t)
  { 
    if(pred(thrust::get<2>(t)))
    {
      RandomAccessIterator dst = output + thrust::get<1>(t);
      thrust::detail::device::dereference(dst) = thrust::get<0>(t);
    }
  }
}; // end scatter_if_functor

} // end namespace detail


template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
  void scatter(InputIterator1 first,
               InputIterator1 last,
               InputIterator2 map,
               RandomAccessIterator output)
{
  // since we're hiding the output inside a functor, its device space will get lost
  // we need to create the zip_iterator with the minimum space of first, map, & output

  typedef typename thrust::iterator_space<InputIterator1>::type       Space1;
  typedef typename thrust::iterator_space<InputIterator2>::type       Space2;
  typedef typename thrust::iterator_space<RandomAccessIterator>::type Space3;

  typedef typename thrust::detail::minimum_space<Space1,Space2>::type Space4;
  typedef typename thrust::detail::minimum_space<Space3,Space4>::type Space;

  typedef thrust::detail::forced_iterator<InputIterator1,Space> forced_iterator;

  // force first to be of the minimum space
  forced_iterator first_forced(first), last_forced(last);

  detail::scatter_functor<RandomAccessIterator> func(output);
  thrust::detail::device::for_each(thrust::make_zip_iterator(thrust::make_tuple(first_forced, map)),
                                   thrust::make_zip_iterator(thrust::make_tuple(last_forced,  map + thrust::distance(first, last))),
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
  // since we're hiding the output inside a functor, its device space will get lost
  // we need to create the zip_iterator with the minimum space of first, map, & output

  typedef typename thrust::iterator_space<InputIterator1>::type       Space1;
  typedef typename thrust::iterator_space<InputIterator2>::type       Space2;
  typedef typename thrust::iterator_space<InputIterator3>::type       Space3;
  typedef typename thrust::iterator_space<RandomAccessIterator>::type Space4;

  typedef typename thrust::detail::minimum_space<Space1,Space2>::type Space5;
  typedef typename thrust::detail::minimum_space<Space3,Space4>::type Space6;

  typedef typename thrust::detail::minimum_space<Space5,Space6>::type Space;

  typedef thrust::detail::forced_iterator<InputIterator1,Space> forced_iterator;

  // force first to be of the minimum space
  forced_iterator first_forced(first), last_forced(last);

  detail::scatter_if_functor<RandomAccessIterator, Predicate> func(output, pred);
  thrust::detail::device::for_each(thrust::make_zip_iterator(thrust::make_tuple(first_forced, map, stencil)),
                                   thrust::make_zip_iterator(thrust::make_tuple(last_forced,  map + thrust::distance(first, last), stencil + thrust::distance(first, last))),
                                   func);
} // end scatter_if()

} // end namespace generic
} // end namespace device
} // end namespace detail
} // end namespace thrust

