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
#include <thrust/system/detail/state.h>

namespace thrust
{
namespace system
{
// put the canonical tag in the same ns as the backend's entry points
namespace cpp
{
namespace detail
{

template<typename Derived> struct state;

template<typename Derived>
  struct state_base
{
  typedef thrust::system::detail::state<Derived> type;
};

template<>
  struct state_base<void>
{
  typedef thrust::system::detail::state<
    state<void>
  > type;
};

template<typename Derived>
  struct state
    : state_base<Derived>::type
{};

typedef state<void> tag;

} // end detail

// alias the tag here
using thrust::system::cpp::detail::tag;

} // end cpp
} // end system

// alias cpp's tag at top-level
namespace cpp
{

using thrust::system::cpp::tag;

} // end cpp
} // end thrust

