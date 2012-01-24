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

/*! \file copy.h
 *  \brief C++ implementations of copy functions.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/cpp/detail/tag.h>
#include <thrust/system/detail/internal/scalar/copy.h>

namespace thrust
{
namespace system
{
namespace cpp
{
namespace detail
{

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(tag,
                      InputIterator first,
                      InputIterator last,
                      OutputIterator result)
{
  return thrust::system::detail::internal::scalar::copy(first, last, result);
}

template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(tag,
                        InputIterator first,
                        Size n,
                        OutputIterator result)
{
  return thrust::system::detail::internal::scalar::copy_n(first, n, result);
}

} // end namespace detail
} // end namespace cpp
} // end namespace system
} // end namespace thrust

