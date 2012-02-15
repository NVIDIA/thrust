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

#include <thrust/detail/backend/for_each.h>
#include <thrust/distance.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace generic
{

template<typename InputIterator,
         typename UnaryFunction>
InputIterator for_each(InputIterator first,
                       InputIterator last,
                       UnaryFunction f)
{
  // all iterators are random access right now, so this is safe
  return thrust::detail::backend::for_each_n(first, thrust::distance(first,last), f);
} // end for_each()

} // end generic
} // end backend
} // end detail
} // end thrust

