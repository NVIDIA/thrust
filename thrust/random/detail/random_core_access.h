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

#pragma once

namespace thrust
{

namespace random
{

namespace detail
{

struct random_core_access
{

template<typename OStream, typename Engine>
static OStream &stream_out(OStream &os, const Engine &e)
{
  return e.stream_out(os);
}

template<typename IStream, typename Engine>
static IStream &stream_in(IStream &is, Engine &e)
{
  return e.stream_in(is);
}

template<typename Engine>
__host__ __device__
static bool equal(const Engine &lhs, const Engine &rhs)
{
  return lhs.equal(rhs);
}

}; // end random_core_access

} // end detail

} // end random

} // end thrust

