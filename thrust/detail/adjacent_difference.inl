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


/*! \file adjacent_difference.inl
 *  \brief Inline file for adjacent_difference.h
 */

#include <thrust/detail/config.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/adjacent_difference.h>
#include <thrust/detail/adl_helper.h>

namespace thrust
{

template <typename InputIterator, typename OutputIterator>
OutputIterator adjacent_difference(InputIterator first, InputIterator last, 
                                   OutputIterator result)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::adjacent_difference;

  typedef typename thrust::iterator_system<InputIterator>::type  system1;
  typedef typename thrust::iterator_system<OutputIterator>::type system2;

  return adjacent_difference(select_system(system1(), system2()), first, last, result);
} // end adjacent_difference()

template <typename InputIterator, typename OutputIterator, typename BinaryFunction>
OutputIterator adjacent_difference(InputIterator first, InputIterator last,
                                   OutputIterator result,
                                   BinaryFunction binary_op)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::adjacent_difference;

  typedef typename thrust::iterator_system<InputIterator>::type  system1;
  typedef typename thrust::iterator_system<OutputIterator>::type system2;

  return adjacent_difference(select_system(system1(), system2()), first, last, result, binary_op);
} // end adjacent_difference()


} // end namespace thrust

