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

// this awkward sequence of definitions arise
// from the desire both for tag to derive
// from dispatchable and for dispatchable
// to convert to tag (when dispatchable is not
// an ancestor of tag)

// forward declaration of tag
struct tag;

// forward declaration of dispatchable
template<typename> struct dispatchable;

// specialize dispatchable for tag
template<>
  struct dispatchable<tag>
    : thrust::dispatchable<tag>
{};

// tag's definition comes before the
// generic definition of dispatchable
struct tag : dispatchable<tag> {};

// allow conversion to tag when it is not a successor
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

// alias dispatchable and tag here
using thrust::system::cpp::detail::dispatchable;
using thrust::system::cpp::detail::tag;

} // end cpp
} // end system

// alias items at top-level
namespace cpp
{

using thrust::system::cpp::dispatchable;
using thrust::system::cpp::tag;

} // end cpp
} // end thrust

