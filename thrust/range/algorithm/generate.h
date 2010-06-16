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

#include <thrust/detail/config.h>
#include <thrust/range/algorithm/for_each.h>

namespace thrust
{

namespace experimental
{

namespace range
{


namespace detail
{

template<typename Generator>
  struct generate_functor
{
  Generator gen;
  
  generate_functor(Generator g)
    : gen(g){}
  
  template<typename T>
  __host__ __device__
  void operator()(T &x)
  {
    x = gen();
  }
} // end generate_functor

} // end detail


template<typename ForwardRange, typename Generator>
  inline ForwardRange &generate(ForwardRange &rng, Generator gen)
{
  for_each(rng, generate_functor<Generator>(gen));
  return rng;
} // end generate()


} // end range

} // end experimental

} // end thrust

