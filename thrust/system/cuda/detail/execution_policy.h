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
#include <thrust/detail/execution_policy.h>
#include <thrust/system/cpp/detail/execution_policy.h>
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
// from execution_policy and for execution_policy
// to convert to tag (when execution_policy is not
// an ancestor of tag)

// forward declaration of tag
struct tag;

// forward declaration of execution_policy
template<typename> struct execution_policy;

// specialize execution_policy for tag
template<>
  struct execution_policy<tag>
    : thrust::execution_policy<tag>
{};

// tag's definition comes before the
// generic definition of execution_policy
struct tag : execution_policy<tag> {};

// allow conversion to tag when it is not a successor
template<typename Derived>
  struct execution_policy
    : thrust::execution_policy<Derived>
{
  // allow conversion to tag
  inline operator tag () const
  {
    return tag();
  }
};


template<typename System1, typename System2>
  struct cross_system
    : thrust::execution_policy<cross_system<System1,System2> >
{
  inline __host__ __device__
  cross_system(thrust::execution_policy<System1> &system1,
               thrust::execution_policy<System2> &system2)
    : system1(system1), system2(system2)
  {}

  thrust::execution_policy<System1> &system1;
  thrust::execution_policy<System2> &system2;

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
cross_system<System1,System2> select_system(const execution_policy<System1> &system1, const thrust::cpp::execution_policy<System2> &system2)
{
  thrust::execution_policy<System1> &non_const_system1 = const_cast<execution_policy<System1>&>(system1);
  thrust::cpp::execution_policy<System2> &non_const_system2 = const_cast<thrust::cpp::execution_policy<System2>&>(system2);
  return cross_system<System1,System2>(non_const_system1,non_const_system2);
}


template<typename System1, typename System2>
inline __host__ __device__
cross_system<System1,System2> select_system(const thrust::cpp::execution_policy<System1> &system1, execution_policy<System2> &system2)
{
  thrust::cpp::execution_policy<System1> &non_const_system1 = const_cast<thrust::cpp::execution_policy<System1>&>(system1);
  thrust::execution_policy<System2> &non_const_system2 = const_cast<execution_policy<System2>&>(system2);
  return cross_system<System1,System2>(non_const_system1,non_const_system2);
}


} // end detail

// alias execution_policy and tag here
using thrust::system::cuda::detail::execution_policy;
using thrust::system::cuda::detail::tag;

} // end cuda
} // end system

// alias items at top-level
namespace cuda
{

using thrust::system::cuda::execution_policy;
using thrust::system::cuda::tag;

} // end cuda
} // end thrust

