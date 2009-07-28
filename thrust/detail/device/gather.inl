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

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/device/cuda/vectorize.h>
#include <thrust/detail/device/dereference.h>

namespace thrust
{

namespace detail
{

namespace device
{

namespace detail
{

//////////////
// Functors //
//////////////
template <typename ForwardIterator, typename InputIterator, typename RandomAccessIterator>
struct gather_functor
{
    ForwardIterator first;
    InputIterator map;
    RandomAccessIterator input;

    gather_functor(ForwardIterator _first, InputIterator _map, RandomAccessIterator _input)
      : first(_first), map(_map), input(_input) {}
  
    template <typename IntegerType>
        __device__
        void operator()(const IntegerType& i)
        { 
            //output[i] = input[map[i]];
            thrust::detail::device::dereference(first, i) = thrust::detail::device::dereference(input, thrust::detail::device::dereference(map, i)); 
        }
}; // end gather_functor

template <typename ForwardIterator, typename InputIterator1, typename InputIterator2, typename RandomAccessIterator, typename Predicate>
struct gather_if_functor
{
    ForwardIterator first;
    InputIterator1 map;
    InputIterator2 stencil;
    RandomAccessIterator input;
    Predicate pred;

    gather_if_functor(ForwardIterator _first, InputIterator1 _map, InputIterator2 _stencil, RandomAccessIterator _input, Predicate _pred)
      : first(_first), map(_map), stencil(_stencil), input(_input), pred(_pred) {}
  
    template <typename IntegerType>
        __device__
        void operator()(const IntegerType& i)
        { 
            //if(pred(stencil[i])
            //    output[i] = input[map[i]];
            if(pred(thrust::detail::device::dereference(stencil, i)))
                thrust::detail::device::dereference(first, i) = thrust::detail::device::dereference(input, thrust::detail::device::dereference(map, i)); 
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
    detail::gather_functor<ForwardIterator, InputIterator, RandomAccessIterator> func(first, map, input);
    thrust::detail::device::cuda::vectorize(last - first, func);
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
    detail::gather_if_functor<ForwardIterator, InputIterator1, InputIterator2, RandomAccessIterator, Predicate> func(first, map, stencil, input, pred);
    thrust::detail::device::cuda::vectorize(last - first, func);
} // end gather_if()


} // end namespace device

} // end namespace detail

} // end namespace thrust

