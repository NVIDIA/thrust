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

namespace thrust
{

namespace detail
{

// an oracle to tell us how to define uint64_t
template<int word_size = sizeof(int*)> struct divine_uint64_t;

// 32b machine type
template<>
  struct divine_uint64_t<4>
{
  typedef unsigned long long int type;
};

// 64b machine type
template<>
  struct divine_uint64_t<8>
{
  typedef unsigned long long int type;
};

typedef unsigned int            uint32_t;
typedef divine_uint64_t<>::type uint64_t;


// an oracle to tell us how to define intptr_t
template<int word_size = sizeof(int*)> struct divine_intptr_t;

// use uint32_t on 32b platforms
template<>
  struct divine_intptr_t<4>
{
  typedef thrust::detail::uint32_t type;
};

// use uint64_t on 64b platforms
template<>
  struct divine_intptr_t<8>
{
  typedef thrust::detail::uint64_t type;
};

typedef divine_intptr_t<>::type   intptr_t;
typedef thrust::detail::intptr_t uintptr_t;


} // end detail

} // end thrust

