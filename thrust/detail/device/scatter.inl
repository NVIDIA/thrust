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
template <typename InputIterator1, typename InputIterator2, typename RandomAccessIterator>
struct scatter_functor
{
    InputIterator1 first;
    InputIterator2 map;
    RandomAccessIterator output;

    scatter_functor(InputIterator1 _first, InputIterator2 _map, RandomAccessIterator _output) 
        : first(_first), map(_map), output(_output) {}
  
    template <typename IntegerType>
        __device__
        void operator()(const IntegerType& i)
        { 
            //output[map[i]] = input[i];
            thrust::detail::device::dereference(output, thrust::detail::device::dereference(map, i)) = thrust::detail::device::dereference(first, i); 
        }
}; // end scatter_functor


template <typename InputIterator1, typename InputIterator2, typename InputIterator3, typename RandomAccessIterator, typename Predicate>
struct scatter_if_functor
{
    InputIterator1 first;
    InputIterator2 map;
    InputIterator3 stencil;
    RandomAccessIterator output;
    Predicate pred;

    scatter_if_functor(InputIterator1 _first, InputIterator2 _map, InputIterator3 _stencil, RandomAccessIterator _output, Predicate _pred)
        : first(_first), map(_map), stencil(_stencil), output(_output), pred(_pred) {}
  
    template <typename IntegerType>
        __device__
        void operator()(const IntegerType& i)
        { 
            //if(pred(stencil[i]))
            //    output[map[i]] = input[i];
            if(pred(thrust::detail::device::dereference(stencil, i)))
                thrust::detail::device::dereference(output, thrust::detail::device::dereference(map, i)) = thrust::detail::device::dereference(first, i); 
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
    detail::scatter_functor<InputIterator1, InputIterator2, RandomAccessIterator> func(first, map, output);
    thrust::detail::device::cuda::vectorize(last - first, func);
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
    detail::scatter_if_functor<InputIterator1, InputIterator2, InputIterator3, RandomAccessIterator, Predicate> func(first, map, stencil, output, pred);
    thrust::detail::device::cuda::vectorize(last - first, func);
} // end scatter_if()


} // end namespace device

} // end namespace detail

} // end namespace thrust

