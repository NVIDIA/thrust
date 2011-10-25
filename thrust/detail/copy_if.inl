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

#include <thrust/detail/config.h>
#include <thrust/detail/copy_if.h>
#include <thrust/detail/backend/generic/select_system.h>
#include <thrust/detail/backend/generic/copy_if.h>
#include <thrust/iterator/iterator_traits.h>

// XXX make the backend-specific versions of copy_if available
// XXX try to eliminate the need for these
#include <thrust/system/cpp/detail/copy_if.h>
#include <thrust/detail/backend/omp/copy_if.h>
#include <thrust/detail/backend/cuda/copy_if.h>

namespace thrust
{


template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator copy_if(InputIterator first,
                         InputIterator last,
                         OutputIterator result,
                         Predicate pred)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::copy_if;

  typedef typename thrust::iterator_space<InputIterator>::type space1;
  typedef typename thrust::iterator_space<OutputIterator>::type space2;

  return copy_if(select_system(space1(),space2()), first, last, result, pred);
} // end copy_if()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator copy_if(InputIterator1 first,
                         InputIterator1 last,
                         InputIterator2 stencil,
                         OutputIterator result,
                         Predicate pred)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::copy_if;

  typedef typename thrust::iterator_space<InputIterator1>::type space1;
  typedef typename thrust::iterator_space<InputIterator2>::type space2;
  typedef typename thrust::iterator_space<OutputIterator>::type space3;

  return copy_if(select_system(space1(),space2(),space3()), first, last, stencil, result, pred);
} // end copy_if()


} // end thrust

