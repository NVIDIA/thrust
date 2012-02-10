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

namespace thrust
{
namespace system
{
// put the canonical tag in the same ns as the backend's entry points
namespace omp
{
namespace detail
{

// omp inherits cpp's functionality
struct tag : thrust::system::cpp::tag {};


// select system overloads
__host__ __device__
inline tag select_system(tag, tag)
{
  return tag();
} // end select_system()

__host__ __device__
inline tag select_system(tag, thrust::any_system_tag)
{
  return tag();
} // end select_system()

__host__ __device__
inline tag select_system(thrust::any_system_tag, tag)
{
  return tag();
} // end select_system()

__host__ __device__
inline tag select_system(tag, thrust::system::cpp::tag)
{
  return tag();
} // end select_system()

__host__ __device__
inline tag select_system(thrust::system::cpp::tag, tag)
{
  return tag();
} // end select_system()


// XXX select_system(tbb, omp) & select_system(omp, tbb) are ambiguous
//     because both convert to cpp without these overloads, which we
//     arbitrarily define in the omp backend
__host__ __device__
inline tag select_system(tag, thrust::system::tbb::tag)
{
  return tag();
} // end select_system()

__host__ __device__
inline tag select_system(thrust::system::tbb::tag, tag)
{
  return tag();
} // end select_system()


} // end detail

// alias the tag here
using thrust::system::omp::detail::tag;

} // end omp
} // end system

// alias omp's tag at top-level
namespace omp
{

using thrust::system::omp::tag;

} // end omp
} // end thrust

