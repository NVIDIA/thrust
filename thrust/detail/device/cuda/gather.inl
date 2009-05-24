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
template <typename OutputType, typename IndexType, typename InputType>
struct gather_functor
{
        OutputType * output;
  const IndexType  * map;
  const InputType  * input;

  gather_functor(OutputType * _output, const IndexType * _map, const InputType * _input) 
      : output(_output), map(_map), input(_input) {}
  
  template <typename IntegerType>
      __host__ __device__
  void operator()(const IntegerType& i) { output[i] = input[map[i]]; }
}; // end gather_functor

template <typename OutputType, typename IndexType, typename StencilType, typename InputType, typename Predicate>
struct gather_if_functor
{
        OutputType  * output;
  const IndexType   * map;
  const StencilType * stencil;
  const InputType   * input;
  Predicate pred;

  gather_if_functor(      OutputType  * _output, 
                    const IndexType   * _map,
                    const StencilType * _stencil,
                    const InputType   * _input,
                    Predicate _pred) 
      : output(_output), map(_map), stencil(_stencil), input(_input), pred(_pred) {}
  
  template <typename IntegerType>
      __host__ __device__
  void operator()(const IntegerType& i) { if (pred(stencil[i])) output[i] = input[map[i]]; }
}; // end gather_if_functor

} // end detail



template<typename ForwardIterator,
         typename InputIterator,
         typename RandomAccessIterator>
  void gather(ForwardIterator first,
              ForwardIterator last,
              InputIterator  map,
              RandomAccessIterator input)
{
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type OutputType;
  typedef typename thrust::iterator_traits<InputIterator>::value_type IndexType;
  typedef typename thrust::iterator_traits<RandomAccessIterator>::value_type InputType;

  // XXX use make_device_dereferenceable here instead of assuming &*first, &*map, &*input are device_ptr
  detail::gather_functor<OutputType,IndexType,InputType> func((&*first).get(), (&*map).get(), (&*input).get());

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
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type OutputType;
  typedef typename thrust::iterator_traits<InputIterator1>::value_type IndexType;
  typedef typename thrust::iterator_traits<InputIterator2>::value_type PredicateType;
  typedef typename thrust::iterator_traits<RandomAccessIterator>::value_type InputType;

  // XXX use make_device_dereferenceable here instead of assuming &*first, &*map, &*stencil, &*input are device_ptr
  detail::gather_if_functor<OutputType,IndexType,PredicateType,InputType,Predicate> func((&*first).get(), (&*map).get(), (&*stencil).get(), (&*input).get(), pred);

  thrust::detail::device::cuda::vectorize(last - first, func);
} // end gather_if()


} // end namespace cuda

} // end namespace device

} // end namespace detail

} // end namespace thrust

