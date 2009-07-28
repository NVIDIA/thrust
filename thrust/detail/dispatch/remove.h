/*
 *  Copyright 2008-2009 NVIDIA Corporation
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

#include <algorithm>
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
  ForwardIterator remove_if(ForwardIterator begin,
                            ForwardIterator end,
                            Predicate pred,
                            thrust::experimental::space::host)
{
  return std::remove_if(begin, end, pred);
}

template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator begin,
                                InputIterator end,
                                OutputIterator result,
                                Predicate pred,
                                thrust::experimental::space::host,
                                thrust::experimental::space::host)
{
  return std::remove_copy_if(begin, end, result, pred);
}


//////////////////
// Device Paths //
//////////////////
template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator remove_if(ForwardIterator begin,
                            ForwardIterator end,
                            Predicate pred,
                            thrust::experimental::space::device)
{
  return thrust::detail::device::remove_if(begin, end, pred);
} 


template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator begin,
                                InputIterator end,
                                OutputIterator result,
                                Predicate pred,
                                thrust::experimental::space::device,
                                thrust::experimental::space::device)
{
  return thrust::detail::device::remove_copy_if(begin, end, result, pred);
}

} // end namespace dispatch

} // end namespace detail

} // end namespace thrust

