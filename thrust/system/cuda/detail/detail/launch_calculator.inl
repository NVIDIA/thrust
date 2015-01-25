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

// do not attempt to compile this file with any other compiler
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#include <thrust/system/cuda/detail/runtime_introspection.h>
#include <thrust/system/cuda/detail/cuda_launch_config.h>
#include <thrust/system/cuda/detail/detail/launch_closure.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace detail
{

template<typename Closure>
__host__ __device__
launch_calculator<Closure>::launch_calculator(void)
  : properties(device_properties()),
    attributes(closure_attributes<Closure>())
{}
  
template<typename Closure>
__host__ __device__
launch_calculator<Closure>::launch_calculator(const device_properties_t& properties, const function_attributes_t& attributes)
  : properties(properties),
    attributes(attributes)
{}

template<typename Closure>
  template<typename UnaryFunction>
__host__ __device__
thrust::pair<size_t, size_t> launch_calculator<Closure>::default_block_configuration(UnaryFunction block_size_to_smem_size) const
{
  // choose a block size
  std::size_t num_threads_per_block = block_size_with_maximum_potential_occupancy(attributes, properties, block_size_to_smem_size);

  // choose a subscription rate
  std::size_t num_blocks_per_multiprocessor = properties.maxThreadsPerMultiProcessor / num_threads_per_block;

  return thrust::make_pair(num_threads_per_block, num_blocks_per_multiprocessor);
}


template<typename Closure>
__host__ __device__
thrust::pair<size_t, size_t> launch_calculator<Closure>::default_block_configuration(void) const
{
  // choose a block size
  std::size_t num_threads_per_block = block_size_with_maximum_potential_occupancy(attributes, properties);

  // choose a subscription rate
  std::size_t num_blocks_per_multiprocessor = properties.maxThreadsPerMultiProcessor / num_threads_per_block;

  return thrust::make_pair(num_threads_per_block, num_blocks_per_multiprocessor);
}

template<typename Closure>
__host__ __device__
thrust::tuple<size_t,size_t,size_t> launch_calculator<Closure>::with_variable_block_size(void) const
{
  thrust::pair<size_t, size_t> config = default_block_configuration();
  return thrust::tuple<size_t,size_t,size_t>(config.second * properties.multiProcessorCount, config.first, 0);
}

template <typename Closure>
  template <typename UnaryFunction>
__host__ __device__
thrust::tuple<size_t,size_t,size_t> launch_calculator<Closure>::with_variable_block_size(UnaryFunction block_size_to_smem_size) const
{
  thrust::pair<size_t, size_t> config = default_block_configuration(block_size_to_smem_size);
  return thrust::tuple<size_t,size_t,size_t>(config.second * properties.multiProcessorCount, config.first, block_size_to_smem_size(config.first));
}
  
template<typename Closure>
__host__ __device__
thrust::tuple<size_t,size_t,size_t> launch_calculator<Closure>::with_variable_block_size_available_smem(void) const
{
  thrust::pair<size_t, size_t> config = default_block_configuration();
  size_t smem_per_block = proportional_smem_allocation(properties, attributes, config.second);
  return thrust::tuple<size_t,size_t,size_t>(config.second * properties.multiProcessorCount, config.first, smem_per_block);
}

} // end detail
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

