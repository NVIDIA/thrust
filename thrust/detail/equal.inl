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


/*! \file equal.inl
 *  \brief Inline file for equal.h.
 */

#include <thrust/equal.h>
#include <thrust/detail/backend/generic/select_system.h>
#include <thrust/detail/backend/generic/equal.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{

template <typename InputIterator1, typename InputIterator2>
bool equal(InputIterator1 first1, InputIterator1 last1,
           InputIterator2 first2)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::equal;

  typedef typename thrust::iterator_space<InputIterator1>::type space1;
  typedef typename thrust::iterator_space<InputIterator2>::type space2;

  return equal(select_system(space1(),space2()), first1, last1, first2);
}

template <typename InputIterator1, typename InputIterator2, 
          typename BinaryPredicate>
bool equal(InputIterator1 first1, InputIterator1 last1,
           InputIterator2 first2, BinaryPredicate binary_pred)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::equal;

  typedef typename thrust::iterator_space<InputIterator1>::type space1;
  typedef typename thrust::iterator_space<InputIterator2>::type space2;

  return equal(select_system(space1(),space2()), first1, last1, first2, binary_pred);
}

} // end namespace thrust

