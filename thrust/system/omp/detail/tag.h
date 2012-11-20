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
#include <thrust/system/cpp/detail/tag.h>
#include <thrust/system/tbb/detail/tag.h>
#include <thrust/iterator/detail/any_system_tag.h>
#include <thrust/detail/type_traits.h>

namespace thrust
{
namespace system
{
// put the canonical tag in the same ns as the backend's entry points
namespace omp
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
    : thrust::system::cpp::detail::dispatchable<tag>
{};

// tag's definition comes before the
// generic definition of dispatchable
struct tag : dispatchable<tag> {};

// allow conversion to tag when it is not a successor
template<typename Derived>
  struct dispatchable
    : thrust::system::cpp::detail::dispatchable<Derived>
{
  // allow conversion to tag
  inline operator tag () const
  {
    return tag();
  }
};


// overloads of select_system

// XXX select_system(tbb, omp) & select_system(omp, tbb) are ambiguous
//     because both convert to cpp without these overloads, which we
//     arbitrarily define in the omp backend

template<typename System1, typename System2>
inline __host__ __device__
  System1 select_system(dispatchable<System1> s, thrust::system::tbb::detail::dispatchable<System2>)
{
  return thrust::detail::derived_cast(s);
} // end select_system()


template<typename System1, typename System2>
inline __host__ __device__
  System2 select_system(thrust::system::tbb::detail::dispatchable<System1>, dispatchable<System2> s)
{
  return thrust::detail::derived_cast(s);
} // end select_system()


} // end detail

// alias dispatchable and tag here
using thrust::system::omp::detail::dispatchable;
using thrust::system::omp::detail::tag;

} // end omp
} // end system

// alias items at top-level
namespace omp
{

using thrust::system::omp::dispatchable;
using thrust::system::omp::tag;

} // end omp
} // end thrust

