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


/*! \file generate.h
 *  \brief Device implementation of generate.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/device/dereference.h>
#include <thrust/detail/device/cuda/vectorize.h>

namespace thrust
{

namespace detail
{

namespace device
{

namespace detail
{

template<typename ForwardIterator,
         typename Generator>
struct generate_functor
{
  ForwardIterator first;
  Generator gen;

  generate_functor(ForwardIterator _first, Generator _gen)
    : first(_first), gen(_gen){}

  template<typename IntegerType>
      __device__
      void operator()(IntegerType i)
      {
          thrust::detail::device::dereference(first, i) = gen();
      }
}; // end generate_functor
  
} // end namespace detail


template<typename ForwardIterator,
         typename Generator>
  void generate(ForwardIterator first,
                ForwardIterator last,
                Generator gen)
{
    detail::generate_functor<ForwardIterator, Generator> f(first, gen);
    thrust::detail::device::cuda::vectorize(last - first, f);
} // end generate()

} // end namespace device

} // end namespace detail

} // end namespace thrust

