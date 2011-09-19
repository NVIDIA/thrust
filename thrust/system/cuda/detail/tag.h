/*
 *  Copyright 2008-2011 NVIDIA Corporation
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

namespace thrust
{
// put the canonical tag in the same ns as the backend's entry points
// XXX cuda's entry points should be under system, not backend
namespace detail
{
namespace backend
{
namespace cuda
{

struct tag {};

} // end cuda
} // end backend
} // end detail

namespace system
{
namespace cuda
{

// alias cuda's tag here
using thrust::detail::backend::cuda::tag;

} // end cuda
} // end system

// alias cuda's tag at top-level
namespace cuda
{

using thrust::system::cuda::tag;

} // end cuda

} // end thrust

