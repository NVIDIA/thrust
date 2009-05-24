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
 *  \brief Defines the interface to the
 *         dispatch layer of the generate function.
 */

#pragma once

#include <thrust/iterator/iterator_categories.h>
#include <thrust/detail/make_device_dereferenceable.h>
#include <algorithm>

#include <thrust/detail/device/cuda/vectorize.h>

namespace thrust
{

namespace detail
{

namespace dispatch
{

// host path
template<typename ForwardIterator,
         typename Generator>
  void generate(ForwardIterator first,
                ForwardIterator last,
                Generator gen,
                thrust::forward_host_iterator_tag)
{
  std::generate(first, last, gen);
} // end generate()


namespace detail
{

template<typename ForwardIterator,
         typename Generator>
struct generator_functor
{
  ForwardIterator iter;
  Generator gen;

  generator_functor(ForwardIterator i, Generator g)
    :iter(i),gen(g){}

  template<typename IntegerType>
  __host__ __device__
  void operator()(IntegerType i)
  {
    iter[i] = gen();
  }
}; // end generator_functor
  
} // end detail


// device path
template<typename ForwardIterator,
         typename Generator>
  void generate(ForwardIterator first,
                ForwardIterator last,
                Generator gen,
                thrust::random_access_device_iterator_tag)
{
  typedef typename thrust::detail::device_dereferenceable_iterator_traits<ForwardIterator>::device_dereferenceable_type Iter;
  typename thrust::iterator_traits<ForwardIterator>::difference_type n = last - first;

  Iter iter = thrust::detail::make_device_dereferenceable<ForwardIterator>::transform(first);

  detail::generator_functor<Iter, Generator> f(iter,gen);
  thrust::detail::device::cuda::vectorize(n, f);
} // end generate()

} // end dispatch

} // end detail

} // end thrust

