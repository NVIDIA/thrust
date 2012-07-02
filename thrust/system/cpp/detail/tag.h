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
#include <thrust/detail/dispatchable.h>

namespace thrust
{
namespace system
{
// put the canonical tag in the same ns as the backend's entry points
namespace cpp
{
namespace detail
{

// forward declaration of dispatchable
template<typename> struct dispatchable;

// tag's specialization comes first
template<>
  struct dispatchable<void>
    : thrust::dispatchable< dispatchable<void> >
{};

// tag is just a typedef for dispatchable<void>
typedef dispatchable<void> tag;

template<typename Derived>
  struct dispatchable
    : thrust::dispatchable<Derived>
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

