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

// forward declaration of state
template<typename> struct state;

// tag's specialization comes first
template<>
  struct state<void>
    : thrust::system::detail::state< state<void> >
{};

// tag is just a typedef for state<void>
typedef state<void> tag;

template<typename Derived>
  struct state
    : thrust::system::detail::state<Derived>
{
  // allow conversion to tag
  inline operator tag () const
  {
    return tag();
  }
};

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

