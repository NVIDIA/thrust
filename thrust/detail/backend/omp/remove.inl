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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/backend/omp/remove.h>
#include <thrust/detail/backend/generic/remove.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace omp
{


template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator remove_if(tag,
                            ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
  // omp prefers generic::remove_if to cpp::remove_if
  return thrust::detail::backend::generic::remove_if(tag(), first, last, pred);
} // end remove_if()


template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
  ForwardIterator remove_if(tag,
                            ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred)
{
  // omp prefers generic::remove_if to cpp::remove_if
  return thrust::detail::backend::generic::remove_if(tag(), first, last, stencil, pred);
} // end remove_if()


template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(tag,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                Predicate pred)
{
  // omp prefers generic::remove_copy_if to cpp::remove_copy_if
  return thrust::detail::backend::generic::remove_copy_if(tag(), first, last, result, pred);
} // end remove_copy_if()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(tag,
                                InputIterator1 first,
                                InputIterator1 last,
                                InputIterator2 stencil,
                                OutputIterator result,
                                Predicate pred)
{
  // omp prefers generic::remove_copy_if to cpp::remove_copy_if
  return thrust::detail::backend::generic::remove_copy_if(tag(), first, last, stencil, result, pred);
} // end remove_copy_if()


} // end namespace omp
} // end namespace backend
} // end namespace detail
} // end namespace thrust

#include <thrust/detail/backend/omp/remove.inl>

