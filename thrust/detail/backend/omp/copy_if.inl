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
#include <thrust/detail/backend/omp/copy_if.h>
#include <thrust/detail/backend/generic/copy_if.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace omp
{


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator copy_if(tag,
                         InputIterator1 first,
                         InputIterator1 last,
                         InputIterator2 stencil,
                         OutputIterator result,
                         Predicate pred)
{
  // omp prefers generic::copy_if to cpp::copy_if
  return thrust::detail::backend::generic::copy_if(tag(), first, last, stencil, result, pred);
} // end copy_if()


} // end namespace omp
} // end backend
} // end detail
} // end thrust

