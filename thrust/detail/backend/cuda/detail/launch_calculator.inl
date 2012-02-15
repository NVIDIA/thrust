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

// do not attempt to compile this file with any other compiler
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#include <thrust/detail/backend/cuda/arch.h>
#include <thrust/detail/backend/cuda/detail/launch_closure.h>

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
launch_calculator<Closure>::launch_calculator(void)
  : properties(thrust::detail::backend::cuda::arch::device_properties()),
    attributes(thrust::detail::backend::cuda::detail::closure_attributes<Closure>())
{}
  
template <typename Closure>
launch_calculator<Closure>::launch_calculator(const arch::device_properties_t& properties, const arch::function_attributes_t& attributes)
  : properties(properties),
    attributes(attributes)
{}

template <typename Closure>
thrust::tuple<size_t,size_t,size_t> launch_calculator<Closure>::with_variable_block_size(void)
{
  thrust::pair<size_t, size_t> config = thrust::detail::backend::cuda::arch::default_block_configuration(properties, attributes);
  return thrust::tuple<size_t,size_t,size_t>(config.second * properties.multiProcessorCount, config.first, 0);
}

template <typename Closure>
  template <typename UnaryFunction>
thrust::tuple<size_t,size_t,size_t> launch_calculator<Closure>::with_variable_block_size(UnaryFunction block_size_to_smem_size)
{
  thrust::pair<size_t, size_t> config = thrust::detail::backend::cuda::arch::default_block_configuration(properties, attributes, block_size_to_smem_size);
  return thrust::tuple<size_t,size_t,size_t>(config.second * properties.multiProcessorCount, config.first, block_size_to_smem_size(config.first));
}
  
template <typename Closure>
thrust::tuple<size_t,size_t,size_t> launch_calculator<Closure>::with_variable_block_size_available_smem(void)
{
  thrust::pair<size_t, size_t> config = thrust::detail::backend::cuda::arch::default_block_configuration(properties, attributes);
  size_t smem_per_block = thrust::detail::backend::cuda::arch::proportional_smem_allocation(properties, attributes, config.second);
  return thrust::tuple<size_t,size_t,size_t>(config.second * properties.multiProcessorCount, config.first, smem_per_block);
}

} // end detail
} // end namespace cuda
} // end namespace backend
} // end namespace detail
} // end namespace thrust

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

