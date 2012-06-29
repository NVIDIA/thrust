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

// forward declaration of state
template<typename> struct state;

// tag's specialization comes first
template<>
  struct state<void>
    : thrust::system::detail::state< state<void> >,
      private virtual thrust::system::detail::final
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

struct cuda_to_cpp  : thrust::system::detail::state<cuda_to_cpp>{};
struct cpp_to_cuda  : thrust::system::detail::state<cpp_to_cuda>{};

// overloads of select_system

// cpp interop
template<typename DerivedSystem1, typename DerivedSystem2>
inline __host__ __device__
cuda_to_cpp select_system(state<DerivedSystem1>, thrust::system::cpp::detail::state<DerivedSystem2>)
{
  return cuda_to_cpp();
}


template<typename DerivedSystem1, typename DerivedSystem2>
inline __host__ __device__
cpp_to_cuda select_system(thrust::system::cpp::detail::state<DerivedSystem1>, state<DerivedSystem2>)
{
  return cpp_to_cuda();
}


} // end detail

// alias the tag here
using thrust::system::cuda::detail::tag;

} // end cuda
} // end system

// alias cuda's tag at top-level
namespace cuda
{

using thrust::system::cuda::tag;

} // end cuda
} // end thrust

