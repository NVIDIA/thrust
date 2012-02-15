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

#include <thrust/detail/backend/cpp/dispatch/copy.h>
#include <thrust/detail/dispatch/is_trivial_copy.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace cpp
{


template<typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result)
{
  return thrust::detail::backend::cpp::dispatch::copy(first, last, result,
      typename thrust::detail::dispatch::is_trivial_copy<InputIterator,OutputIterator>::type());
} // end copy()

template<typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result)
{
  return thrust::detail::backend::cpp::dispatch::copy_n(first, n, result,
      typename thrust::detail::dispatch::is_trivial_copy<InputIterator,OutputIterator>::type());
} // end copy()


} // end cpp
} // end backend
} // end detail
} // end thrust

