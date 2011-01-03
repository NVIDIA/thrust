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

namespace thrust
{

namespace detail
{

namespace device
{

namespace cuda
{

namespace detail
{

template<typename NullaryFunction>
  size_t block_size_with_maximal_occupancy(size_t dynamic_smem_bytes_per_thread = 0);

template<typename NullaryFunction, typename Size1, typename Size2>
  size_t num_blocks_with_maximal_occupancy(Size1 n, Size2 block_size, size_t dynamic_smem_bytes_per_block = 0);

template<typename NullaryFunction, typename Size>
  void launch_closure(NullaryFunction f, Size n);

template<typename NullaryFunction, typename Size1, typename Size2>
  void launch_closure(NullaryFunction f, Size1 num_blocks, Size2 block_size);

template<typename NullaryFunction, typename Size1, typename Size2, typename Size3>
  void launch_closure(NullaryFunction f, Size1 num_blocks, Size2 block_size, Size3 smem_size);

} // end detail

} // end cuda
  
} // end device
  
} // end detail

} // end thrust

#include <thrust/detail/device/cuda/detail/launch_closure.inl>

