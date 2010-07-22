/*
 *  Copyright 2008-2010 NVIDIA Corporation
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

//  Copyright Neil Groves 2009. Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//
// For more information, see http://www.boost.org/libs/range/

#pragma once

#include <thrust/range/begin.h>
#include <thrust/range/end.h>
#include <thrust/for_each.h>

namespace thrust
{

namespace experimental
{

namespace range
{


template<typename SinglePassRange, typename UnaryFunction>
  inline void for_each(SinglePassRange &rng, UnaryFunction f)
{
  return thrust::for_each(begin(rng), end(rng), f);
} // end for_each()


template<typename SinglePassRange, typename UnaryFunction>
  inline void for_each(const SinglePassRange &rng, UnaryFunction f)
{
  return thrust::for_each(begin(rng), end(rng), f);
} // end for_each()


} // end range

} // end experimental

} // end thrust

