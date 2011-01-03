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


/*! \file remove.h
 *  \brief Dispatch layer to the remove functions.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/host/remove.h>
#include <thrust/detail/device/remove.h>

namespace thrust
{
namespace detail
{
namespace dispatch
{

////////////////
// Host Paths //
////////////////
template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator remove_if(ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred,
                            thrust::host_space_tag)
{
  return thrust::detail::host::remove_if(first, last, pred);
}

template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
  ForwardIterator remove_if(ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred,
                            thrust::host_space_tag,
                            thrust::host_space_tag)
{
  return thrust::detail::host::remove_if(first, last, stencil, pred);
}

template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                Predicate pred,
                                thrust::host_space_tag,
                                thrust::host_space_tag)
{
  // Note: this version can't be implemented by passing 'first' as the stencil
  // argument of function below as it would violate InputIterator's semantics.
  return thrust::detail::host::remove_copy_if(first, last, result, pred);
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator1 first,
                                InputIterator1 last,
                                InputIterator2 stencil,
                                OutputIterator result,
                                Predicate pred,
                                thrust::host_space_tag,
                                thrust::host_space_tag,
                                thrust::host_space_tag)
{
  return thrust::detail::host::remove_copy_if(first, last, stencil, result, pred);
}


//////////////////
// Device Paths //
//////////////////
template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator remove_if(ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred,
                            thrust::device_space_tag)
{
  return thrust::detail::device::remove_if(first, last, pred);
}

template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
  ForwardIterator remove_if(ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred,
                            thrust::device_space_tag,
                            thrust::device_space_tag)
{
  return thrust::detail::device::remove_if(first, last, stencil, pred);
} 

template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                Predicate pred,
                                thrust::device_space_tag,
                                thrust::device_space_tag)
{
  // Note: this version can't be implemented by passing 'first' as the stencil
  // argument of function below as it would violate InputIterator's semantics.
  return thrust::detail::device::remove_copy_if(first, last, result, pred);
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator1 first,
                                InputIterator1 last,
                                InputIterator2 stencil,
                                OutputIterator result,
                                Predicate pred,
                                thrust::device_space_tag,
                                thrust::device_space_tag,
                                thrust::device_space_tag)
{
  return thrust::detail::device::remove_copy_if(first, last, stencil, result, pred);
}

} // end namespace dispatch
} // end namespace detail
} // end namespace thrust

