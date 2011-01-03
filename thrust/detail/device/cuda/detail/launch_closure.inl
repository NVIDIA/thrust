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

// do not attempt to compile this file with any other compiler
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#include <algorithm>

#include <thrust/detail/device/cuda/arch.h>
#include <thrust/detail/device/cuda/malloc.h>
#include <thrust/detail/device/cuda/free.h>
#include <thrust/detail/device/cuda/synchronize.h>

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
__global__
void launch_closure_by_value(NullaryFunction f)
{
  f();
}


template<typename NullaryFunction>
__global__
void launch_closure_by_pointer(const NullaryFunction *f)
{
  // copy to registers
  NullaryFunction f_reg = *f;
  f_reg();
}


template<typename NullaryFunction,
         bool launch_by_value = sizeof(NullaryFunction) <= 256>
  struct closure_launcher_base
{
  inline static size_t block_size_with_maximal_occupancy(size_t dynamic_smem_bytes_per_thread = 0)
  {
    return thrust::detail::device::cuda::arch::max_blocksize_with_highest_occupancy(detail::launch_closure_by_value<NullaryFunction>, dynamic_smem_bytes_per_thread);
  }

  template<typename Size1, typename Size2>
  static size_t num_blocks_with_maximal_occupancy(Size1 n, Size2 block_size, size_t dynamic_smem_bytes_per_block)
  {
    const size_t max_blocks = thrust::detail::device::cuda::arch::max_active_blocks(detail::launch_closure_by_value<NullaryFunction>, block_size, dynamic_smem_bytes_per_block);
    return (std::min)(max_blocks, ( n + (block_size - 1) ) / block_size);
  }

  template<typename Size1, typename Size2, typename Size3>
  static void launch(NullaryFunction f, Size1 num_blocks, Size2 block_size, Size3 smem_size)
  {
    detail::launch_closure_by_value<<<(unsigned int) num_blocks, (unsigned int) block_size, (unsigned int) smem_size>>>(f);
    synchronize_if_enabled("launch_closure_by_value");
  }
}; // end closure_launcher_base


template<typename NullaryFunction>
  struct closure_launcher_base<NullaryFunction,false>
{
  inline static size_t block_size_with_maximal_occupancy(size_t dynamic_smem_bytes_per_thread = 0)
  {
    return thrust::detail::device::cuda::arch::max_blocksize_with_highest_occupancy(detail::launch_closure_by_pointer<NullaryFunction>);
  }

  template<typename Size1, typename Size2>
  static size_t num_blocks_with_maximal_occupancy(Size1 n, Size2 block_size, size_t dynamic_smem_bytes_per_block)
  {
    const size_t max_blocks = thrust::detail::device::cuda::arch::max_active_blocks(detail::launch_closure_by_pointer<NullaryFunction>, block_size, dynamic_smem_bytes_per_block);
    return (std::min)(max_blocks, ( n + (block_size - 1) ) / block_size);
  }

  template<typename Size1, typename Size2, typename Size3>
  static void launch(NullaryFunction f, Size1 num_blocks, Size2 block_size, Size3 smem_size)
  {
    // allocate device memory for the argument
    thrust::device_ptr<void> temp_ptr = thrust::detail::device::cuda::malloc<0>(sizeof(NullaryFunction));

    // cast to NullaryFunction *
    thrust::device_ptr<NullaryFunction> f_ptr(reinterpret_cast<NullaryFunction*>(temp_ptr.get()));

    // copy
    *f_ptr = f;

    // launch
    detail::launch_closure_by_pointer<<<(unsigned int) num_blocks, (unsigned int) block_size, (unsigned int) smem_size>>>(f_ptr.get());
    synchronize_if_enabled("launch_closure_by_pointer");

    // free device memory
    thrust::detail::device::cuda::free<0>(f_ptr);
  }
};


template<typename NullaryFunction>
  struct closure_launcher
    : public closure_launcher_base<NullaryFunction>
{
  typedef closure_launcher_base<NullaryFunction> super_t;

  template<typename Size>
  static thrust::pair<size_t,size_t> configuration_with_maximal_occupancy(Size n)
  {
    const size_t block_size = super_t::block_size_with_maximal_occupancy();
    const size_t num_blocks = super_t::num_blocks_with_maximal_occupancy(n, block_size, 0u);

    return thrust::make_pair(num_blocks, block_size);
  }

  template<typename Size>
  static void launch(NullaryFunction f, Size n)
  {
    thrust::pair<size_t, size_t> config = configuration_with_maximal_occupancy(n);
    super_t::launch(f, config.first, config.second, 0u);
  }

  template<typename Size1, typename Size2, typename Size3>
  static void launch(NullaryFunction f, Size1 num_blocks, Size2 block_size, Size3 smem_size)
  {
    super_t::launch(f,num_blocks,block_size,smem_size);
  }
};


template<typename NullaryFunction>
  size_t block_size_with_maximal_occupancy(size_t dynamic_smem_bytes_per_thread)
{
  return closure_launcher<NullaryFunction>::block_size_with_maximal_occupancy(dynamic_smem_bytes_per_thread);
} // end block_size_with_maximal_occupancy()

template<typename NullaryFunction, typename Size1, typename Size2>
  size_t num_blocks_with_maximal_occupancy(Size1 n, Size2 block_size, size_t dynamic_smem_bytes_per_block)
{
  return closure_launcher<NullaryFunction>::num_blocks_with_maximal_occupancy(n, block_size, dynamic_smem_bytes_per_block);
} // end num_blocks_with_maximal_occupancy()

template<typename NullaryFunction, typename Size>
  void launch_closure(NullaryFunction f, Size n)
{
  closure_launcher<NullaryFunction>::launch(f,n);
} // end launch_closure()

template<typename NullaryFunction, typename Size1, typename Size2>
  void launch_closure(NullaryFunction f, Size1 num_blocks, Size2 block_size)
{
  launch_closure(f, num_blocks, block_size, 0u);
} // end launch_closure()

template<typename NullaryFunction, typename Size1, typename Size2, typename Size3>
  void launch_closure(NullaryFunction f, Size1 num_blocks, Size2 block_size, Size3 smem_size)
{
  closure_launcher<NullaryFunction>::launch(f, num_blocks, block_size, smem_size);
} // end launch_closure()


} // end detail
} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

