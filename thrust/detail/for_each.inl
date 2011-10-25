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


/*! \file for_each.inl
 *  \brief Inline file for for_each.h.
 */

#include <thrust/detail/config.h>
#include <thrust/for_each.h>
#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/backend/generic/select_system.h>
#include <thrust/detail/backend/generic/for_each.h>

// XXX make the backend-specific versions available
// XXX try to eliminate the need for these
#include <thrust/system/cpp/detail/for_each.h>
#include <thrust/system/omp/detail/for_each.h>
#include <thrust/detail/backend/cuda/for_each.h>

namespace thrust
{

template<typename InputIterator,
         typename UnaryFunction>
InputIterator for_each(InputIterator first,
                       InputIterator last,
                       UnaryFunction f)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::for_each;

  typedef typename thrust::iterator_space<InputIterator>::type space;

  return for_each(select_system(space()), first, last, f);
} // end for_each()

template<typename InputIterator,
         typename Size,
         typename UnaryFunction>
InputIterator for_each_n(InputIterator first,
                         Size n,
                         UnaryFunction f)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::for_each_n;

  typedef typename thrust::iterator_space<InputIterator>::type space;

  return for_each_n(select_system(space()), first, n, f);
} // end for_each_n()

} // end namespace thrust

