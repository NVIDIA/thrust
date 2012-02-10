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

struct tag {};

struct cuda_to_cpp  {};
struct cpp_to_cuda  {};


__host__ __device__
inline tag select_system(tag, tag)
{
  return tag();
}

__host__ __device__
inline tag select_system(tag, thrust::any_system_tag)
{
  return tag();
}

__host__ __device__
inline tag select_system(thrust::any_system_tag, tag)
{
  return tag();
}

__host__ __device__
inline cuda_to_cpp select_system(tag, thrust::system::cpp::tag)
{
  return cuda_to_cpp();
}

__host__ __device__
inline cpp_to_cuda select_system(thrust::system::cpp::tag, tag)
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

