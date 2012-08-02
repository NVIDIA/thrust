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
#include <thrust/system/cpp/detail/tag.h>
#include <thrust/iterator/detail/any_system_tag.h>

namespace thrust
{
namespace system
{
namespace cuda
{
// put the canonical tag in the same ns as the backend's entry points
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


template<typename System1, typename System2>
  struct cross_system
    : thrust::dispatchable<cross_system<System1,System2> >
{
  inline __host__ __device__
  cross_system(thrust::dispatchable<System1> &system1,
               thrust::dispatchable<System2> &system2)
    : system1(system1), system2(system2)
  {}

  thrust::dispatchable<System1> &system1;
  thrust::dispatchable<System2> &system2;

  inline __host__ __device__
  cross_system<System2,System1> rotate() const
  {
    return cross_system<System2,System1>(system2,system1);
  }
};


// overloads of select_system

// cpp interop
template<typename System1, typename System2>
inline __host__ __device__
cross_system<System1,System2> select_system(const dispatchable<System1> &system1, const thrust::cpp::dispatchable<System2> &system2)
{
  thrust::dispatchable<System1> &non_const_system1 = const_cast<dispatchable<System1>&>(system1);
  thrust::cpp::dispatchable<System2> &non_const_system2 = const_cast<thrust::cpp::dispatchable<System2>&>(system2);
  return cross_system<System1,System2>(non_const_system1,non_const_system2);
}


template<typename System1, typename System2>
inline __host__ __device__
cross_system<System1,System2> select_system(const thrust::cpp::dispatchable<System1> &system1, dispatchable<System2> &system2)
{
  thrust::cpp::dispatchable<System1> &non_const_system1 = const_cast<thrust::cpp::dispatchable<System1>&>(system1);
  thrust::dispatchable<System2> &non_const_system2 = const_cast<dispatchable<System2>&>(system2);
  return cross_system<System1,System2>(non_const_system1,non_const_system2);
}


} // end detail

// alias dispatchable and tag here
using thrust::system::cuda::detail::dispatchable;
using thrust::system::cuda::detail::tag;

} // end cuda
} // end system

// alias items at top-level
namespace cuda
{

using thrust::system::cuda::dispatchable;
using thrust::system::cuda::tag;

} // end cuda
} // end thrust

