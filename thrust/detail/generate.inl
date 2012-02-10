/*
 *  Copyright 2008-2012 NVIDIA Corporation
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


/*! \file generate.inl
 *  \author Jared Hoberock
 *  \brief Inline file for generate.h.
 */

#include <thrust/generate.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/generate.h>
#include <thrust/detail/adl_helper.h>

namespace thrust
{

template<typename ForwardIterator,
         typename Generator>
  void generate(ForwardIterator first,
                ForwardIterator last,
                Generator gen)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::generate;

  typedef typename thrust::iterator_system<ForwardIterator>::type type;

  return generate(select_system(type()), first, last, gen);
} // end generate()


template<typename OutputIterator,
         typename Size,
         typename Generator>
  OutputIterator generate_n(OutputIterator first,
                            Size n,
                            Generator gen)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::generate_n;

  typedef typename thrust::iterator_system<OutputIterator>::type type;

  return generate_n(select_system(type()), first, n, gen);
} // end generate_n()

} // end thrust

