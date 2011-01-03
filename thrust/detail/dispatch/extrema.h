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


/*! \file extrema.h
 *  \brief Dispatch layers for the min_element, max_element and minmax_element functions.
 */

#pragma once

#include <thrust/iterator/iterator_categories.h>

#include <thrust/detail/host/extrema.h>
#include <thrust/detail/device/extrema.h>

namespace thrust
{
namespace detail
{
namespace dispatch
{

////////////////
// Host Paths //
////////////////
template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator min_element(ForwardIterator first, 
                            ForwardIterator last,
                            BinaryPredicate comp, 
                            thrust::host_space_tag)
{
  return thrust::detail::host::min_element(first, last, comp);
} // end min_element()


template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator max_element(ForwardIterator first, 
                            ForwardIterator last,
                            BinaryPredicate comp, 
                            thrust::host_space_tag)
{
  return thrust::detail::host::max_element(first, last, comp);
} // end max_element()


template <typename ForwardIterator, typename BinaryPredicate>
thrust::pair<ForwardIterator,ForwardIterator> minmax_element(ForwardIterator first, 
                                                             ForwardIterator last,
                                                             BinaryPredicate comp, 
                                                             thrust::host_space_tag)
{
  return thrust::detail::host::minmax_element(first, last, comp);
} // end minmax_element()

//////////////////
// Device Paths //
//////////////////
template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator min_element(ForwardIterator first, ForwardIterator last,
                            BinaryPredicate comp, 
                            thrust::device_space_tag)
{
  return thrust::detail::device::min_element(first, last, comp);
} // end min_element()

template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator max_element(ForwardIterator first, ForwardIterator last,
                            BinaryPredicate comp, 
                            thrust::device_space_tag)
{
  return thrust::detail::device::max_element(first, last, comp);
} // end max_element()

template <typename ForwardIterator, typename BinaryPredicate>
thrust::pair<ForwardIterator,ForwardIterator> minmax_element(ForwardIterator first, 
                                                             ForwardIterator last,
                                                             BinaryPredicate comp, 
                                                             thrust::device_space_tag)
{
  return thrust::detail::device::minmax_element(first, last, comp);
} // end minmax_element()


} // end namespace dispatch
} // end namespace detail
} // end namespace thrust

