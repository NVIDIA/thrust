/*
 *  Copyright 2008-2013 NVIDIA Corporation
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
#include <thrust/system/tbb/detail/execution_policy.h>
#include <thrust/detail/execute_with_allocator.h>

namespace thrust
{
namespace system
{
namespace tbb
{
namespace detail
{


struct par_t : thrust::system::tbb::detail::execution_policy<par_t>
{
  par_t() : thrust::system::tbb::detail::execution_policy<par_t>() {}

  template<typename Allocator>
    thrust::detail::execute_with_allocator<Allocator, thrust::system::tbb::detail::execution_policy>
      operator()(Allocator &alloc) const
  {
    return thrust::detail::execute_with_allocator<Allocator, thrust::system::tbb::detail::execution_policy>(alloc);
  }
};


} // end detail


static const detail::par_t par;


} // end tbb
} // end system


// alias par here
namespace tbb
{


using thrust::system::tbb::par;


} // end tbb
} // end thrust

