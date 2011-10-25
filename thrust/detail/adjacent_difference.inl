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


/*! \file adjacent_difference.inl
 *  \brief Inline file for adjacent_difference.h
 */

#include <thrust/detail/config.h>
#include <thrust/detail/backend/generic/select_system.h>
#include <thrust/detail/backend/generic/adjacent_difference.h>

// XXX make the backend-specific versions available
// XXX try to eliminate the need for these
#include <thrust/system/cpp/detail/adjacent_difference.h>
#include <thrust/system/omp/detail/adjacent_difference.h>
#include <thrust/detail/backend/cuda/adjacent_difference.h>

namespace thrust
{

template <typename InputIterator, typename OutputIterator>
OutputIterator adjacent_difference(InputIterator first, InputIterator last, 
                                   OutputIterator result)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::adjacent_difference;

  typedef typename thrust::iterator_space<InputIterator>::type  space1;
  typedef typename thrust::iterator_space<OutputIterator>::type space2;

  return adjacent_difference(select_system(space1(), space2()), first, last, result);
} // end adjacent_difference()

template <typename InputIterator, typename OutputIterator, typename BinaryFunction>
OutputIterator adjacent_difference(InputIterator first, InputIterator last,
                                   OutputIterator result,
                                   BinaryFunction binary_op)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::adjacent_difference;

  typedef typename thrust::iterator_space<InputIterator>::type  space1;
  typedef typename thrust::iterator_space<OutputIterator>::type space2;

  return adjacent_difference(select_system(space1(), space2()), first, last, result, binary_op);
} // end adjacent_difference()


} // end namespace thrust

