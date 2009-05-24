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

namespace thrust
{

namespace detail
{

namespace device
{

namespace cuda
{


namespace detail
{

//////////////
// Functors //
//////////////
template <typename InputType, typename IndexType, typename OutputType>
struct scatter_functor
{
  const InputType  * input;
  const IndexType  * map;
        OutputType * output;

  scatter_functor(const InputType * _input, const IndexType * _map, OutputType * _output) 
      : input(_input), map(_map), output(_output) {}
  
  template <typename IntegerType>
      __host__ __device__
  void operator()(const IntegerType& i) { output[map[i]] = input[i]; }
}; // end scatter_functor


template <typename InputType, typename IndexType, typename StencilType, typename OutputType, typename Predicate>
struct scatter_if_functor
{
  const InputType   * input;
  const IndexType   * map;
  const StencilType * stencil;
        OutputType  * output;
  Predicate pred;

  scatter_if_functor(const InputType   * _input,
                     const IndexType   * _map,
                     const StencilType * _stencil,
                           OutputType  * _output, 
                     Predicate _pred) 
      : input(_input), map(_map), stencil(_stencil), output(_output), pred(_pred) {}
  
  template <typename IntegerType>
      __host__ __device__
  void operator()(const IntegerType i) { if (pred(stencil[i])) output[map[i]] = input[i]; }
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
  typedef typename thrust::iterator_traits<InputIterator1>::value_type InputType;
  typedef typename thrust::iterator_traits<InputIterator2>::value_type IndexType;
  typedef typename thrust::iterator_traits<RandomAccessIterator>::value_type OutputType;

  // XXX use make_device_dereferenceable here instead of assuming &*first, &*map, & &*output are device_ptr
  detail::scatter_functor<InputType,IndexType,OutputType> func((&*first).get(), (&*map).get(), (&*output).get());

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
  typedef typename thrust::iterator_traits<InputIterator1>::value_type InputType;
  typedef typename thrust::iterator_traits<InputIterator2>::value_type IndexType;
  typedef typename thrust::iterator_traits<InputIterator3>::value_type StencilType;
  typedef typename thrust::iterator_traits<RandomAccessIterator>::value_type   OutputType;

  // XXX use make_device_dereferenceable here instead of assuming &*first, &*map, &*stencil, &*output is device_ptr
  detail::scatter_if_functor<InputType,IndexType,StencilType,OutputType,Predicate> func((&*first).get(), (&*map).get(), (&*stencil).get(), (&*output).get(), pred);

  thrust::detail::device::cuda::vectorize(last - first, func);
} // end scatter_if()


} // end namespace cuda

} // end namespace device

} // end namespace detail

} // end namespace thrust

