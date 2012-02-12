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


/*! \file fill.inl
 *  \brief Inline file for fill.h.
 */

#include <thrust/fill.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/fill.h>
#include <thrust/detail/adl_helper.h>

namespace thrust
{

template<typename ForwardIterator, typename T>
  void fill(ForwardIterator first,
            ForwardIterator last,
            const T &value)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::fill;

  typedef typename thrust::iterator_system<ForwardIterator>::type system;

  fill(select_system(system()), first, last, value);
} // end fill()

template<typename OutputIterator, typename Size, typename T>
  OutputIterator fill_n(OutputIterator first,
                        Size n,
                        const T &value)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::fill_n;

  typedef typename thrust::iterator_system<OutputIterator>::type system;

  return fill_n(select_system(system()), first, n, value);
} // end fill()

} // end namespace thrust

