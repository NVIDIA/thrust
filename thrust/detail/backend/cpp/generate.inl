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

#include <thrust/detail/config.h>
#include <thrust/detail/backend/cpp/generate.h>
#include <thrust/detail/backend/cpp/for_each.h>
#include <thrust/detail/internal_functional.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace cpp
{

template<typename OutputIterator,
         typename Size,
         typename Generator>
  OutputIterator generate_n(tag,
                            OutputIterator first,
                            Size n,
                            Generator gen)
{
  return detail::for_each_n(first, n, typename thrust::detail::generate_functor<tag,Generator>::type(gen));
} // end generate_n()

} // end cpp
} // end backend
} // end detail
} // end thrust

