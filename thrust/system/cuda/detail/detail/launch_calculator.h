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

#include <thrust/system/cuda/detail/cuda_launch_config.h>
#include <thrust/tuple.h>

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

template <typename Closure>
class launch_calculator
{
  device_properties_t   properties;
  function_attributes_t attributes;

  public:
  
  launch_calculator(void);

  launch_calculator(const device_properties_t& properties, const function_attributes_t& attributes);

  thrust::tuple<size_t,size_t,size_t> with_variable_block_size(void) const;

  template <typename UnaryFunction>
  thrust::tuple<size_t,size_t,size_t> with_variable_block_size(UnaryFunction block_size_to_smem_size) const;
  
  thrust::tuple<size_t,size_t,size_t> with_variable_block_size_available_smem(void) const;

  private:

  /*! Returns a pair (num_threads_per_block, num_blocks_per_multiprocessor)
   *  where num_threads_per_block is a valid block size for an instance of Closure
   *  chosen by a heuristic and num_blocks_per_multiprocessor is the maximum
   *  number of such blocks that can execute on a streaming multiprocessor at once.
   */
  thrust::pair<size_t, size_t> default_block_configuration() const;

  /*! Returns a pair (num_threads_per_block, num_blocks_per_multiprocessor)
   *  where num_threads_per_block is a valid block size for an instance of Closure
   *  chosen by a heuristic and num_blocks_per_multiprocessor is the maximum
   *  number of such blocks that can execute on a streaming multiprocessor at once.
   *
   *  \param block_size_to_smem_size Mapping from num_threads_per_block to number of
   *                                 dynamically-allocated bytes of shared memory
   */
  template<typename UnaryFunction>
  thrust::pair<size_t, size_t> default_block_configuration(UnaryFunction block_size_to_smem_size) const;
};

} // end namespace detail
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

#include <thrust/system/cuda/detail/detail/launch_calculator.inl>

