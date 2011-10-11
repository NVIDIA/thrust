/*
 *  Copyright 2008-2011 NVIDIA Corporation
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


/*! \file sequence.inl
 *  \brief Inline file for sequence.h.
 */

#include <thrust/detail/config.h>
#include <thrust/sequence.h>
#include <thrust/detail/backend/generic/select_system.h>
#include <thrust/detail/backend/generic/sequence.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{


template<typename ForwardIterator>
  void sequence(ForwardIterator first,
                ForwardIterator last)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::sequence;

  typedef typename thrust::iterator_space<ForwardIterator>::type space;

  return sequence(select_system(space()), first, last);
} // end sequence()


template<typename ForwardIterator, typename T>
  void sequence(ForwardIterator first,
                ForwardIterator last,
                T init)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::sequence;

  typedef typename thrust::iterator_space<ForwardIterator>::type space;

  return sequence(select_system(space()), first, last, init);
} // end sequence()


template<typename ForwardIterator, typename T>
  void sequence(ForwardIterator first,
                ForwardIterator last,
                T init,
                T step)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::sequence;

  typedef typename thrust::iterator_space<ForwardIterator>::type space;

  return sequence(select_system(space()), first, last, init, step);
} // end sequence()


} // end namespace thrust

