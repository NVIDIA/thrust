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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/cpp/detail/tag.h>
#include <thrust/system/detail/internal/scalar/remove.h>

namespace thrust
{
namespace system
{
namespace cpp
{
namespace detail
{

template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator remove_if(tag,
                            ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
  return thrust::system::detail::internal::scalar::remove_if(first, last, pred);
}


template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
  ForwardIterator remove_if(tag,
                            ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred)
{
  return thrust::system::detail::internal::scalar::remove_if(first, last, stencil, pred);
}


template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(tag,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                Predicate pred)
{
  return thrust::system::detail::internal::scalar::remove_copy_if(first, last, result, pred);
}



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
  return thrust::system::detail::internal::scalar::remove_copy_if(first, last, stencil, result, pred);
}

} // end namespace detail
} // end namespace cpp
} // end namespace system
} // end namespace thrust

