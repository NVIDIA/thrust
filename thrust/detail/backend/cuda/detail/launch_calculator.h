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

#include <thrust/detail/config.h>

#include <thrust/detail/backend/cuda/arch.h>
#include <thrust/tuple.h>

// avoid #including a header,
// just provide forward declarations
struct cudaFuncAttributes;

namespace thrust
{
namespace detail
{
namespace backend
{
namespace cuda
{
namespace detail
{

template <typename Closure>
class launch_calculator
{
  arch::device_properties_t properties;
  const cudaFuncAttributes& attributes;

  public:
  
  launch_calculator(void);

  launch_calculator(const arch::device_properties_t& properties, const cudaFuncAttributes& attributes);

  thrust::tuple<size_t,size_t,size_t> with_variable_block_size(void);

  template <typename UnaryFunction>
  thrust::tuple<size_t,size_t,size_t> with_variable_block_size(UnaryFunction block_size_to_smem_size);
  
  thrust::tuple<size_t,size_t,size_t> with_variable_block_size_available_smem(void);
};

} // end namespace detail
} // end namespace cuda
} // end namespace backend
} // end namespace detail
} // end namespace thrust

#include <thrust/detail/backend/cuda/detail/launch_calculator.inl>

