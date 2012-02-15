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
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/backend/dispatch/uninitialized_copy.h>

namespace thrust
{

namespace detail
{

namespace backend
{

template<typename ForwardIterator,
         typename OutputIterator>
  OutputIterator uninitialized_copy(ForwardIterator first,
                                    ForwardIterator last,
                                    OutputIterator result)
{
  typedef typename iterator_traits<ForwardIterator>::value_type ResultType;

  typedef typename thrust::detail::has_trivial_copy_constructor<ResultType>::type ResultTypeHasTrivialCopyConstructor;

  return thrust::detail::backend::dispatch::uninitialized_copy(first, last, result, ResultTypeHasTrivialCopyConstructor());
} // end uninitialized_copy()

} // end backend

} // end detail

} // end thrust

