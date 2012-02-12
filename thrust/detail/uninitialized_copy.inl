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


/*! \file uninitialized_copy.inl
 *  \brief Inline file for uninitialized_copy.h.
 */

#include <thrust/uninitialized_copy.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/uninitialized_copy.h>
#include <thrust/detail/adl_helper.h>

namespace thrust
{


template<typename InputIterator,
         typename ForwardIterator>
  ForwardIterator uninitialized_copy(InputIterator first,
                                     InputIterator last,
                                     ForwardIterator result)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::uninitialized_copy;

  typedef typename thrust::iterator_system<InputIterator>::type   system1;
  typedef typename thrust::iterator_system<ForwardIterator>::type system2;

  return uninitialized_copy(select_system(system1(),system2()), first, last, result);
} // end uninitialized_copy()


template<typename InputIterator,
         typename Size,
         typename ForwardIterator>
  ForwardIterator uninitialized_copy_n(InputIterator first,
                                       Size n,
                                       ForwardIterator result)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::uninitialized_copy_n;

  typedef typename thrust::iterator_system<InputIterator>::type   system1;
  typedef typename thrust::iterator_system<ForwardIterator>::type system2;

  return uninitialized_copy_n(select_system(system1(),system2()), first, n, result);
} // end uninitialized_copy_n()


} // end thrust


