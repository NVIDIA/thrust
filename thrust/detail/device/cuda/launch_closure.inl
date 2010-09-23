/*
 *  Copyright 2008-2010 NVIDIA Corporation
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
  struct closure_launcher
{
  template<typename Size>
  static void launch(NullaryFunction f, Size n)
  {
    const size_t block_size = thrust::detail::device::cuda::arch::max_blocksize_with_highest_occupancy(detail::launch_closure_by_value<NullaryFunction>);
    const size_t max_blocks = thrust::detail::device::cuda::arch::max_active_blocks(detail::launch_closure_by_value<NullaryFunction>, block_size, 0);
    const size_t num_blocks = (std::min)(max_blocks, ( n + (block_size - 1) ) / block_size);

    detail::launch_closure_by_value<<<(unsigned int) num_blocks, (unsigned int) block_size>>>(f);
  }
};

template<typename NullaryFunction>
  struct closure_launcher<NullaryFunction,false>
{
  template<typename Size>
  static void launch(NullaryFunction f, Size n)
  {
    const size_t block_size = thrust::detail::device::cuda::arch::max_blocksize_with_highest_occupancy(detail::launch_closure_by_pointer<NullaryFunction>);
    const size_t max_blocks = thrust::detail::device::cuda::arch::max_active_blocks(detail::launch_closure_by_pointer<NullaryFunction>, block_size, 0);
    const size_t num_blocks = (std::min)(max_blocks, ( n + (block_size - 1) ) / block_size);

    // allocate device memory for the argument
    thrust::device_ptr<void> temp_ptr = thrust::detail::device::cuda::malloc<0>(sizeof(NullaryFunction));

    // cast to NullaryFunction *
    thrust::device_ptr<NullaryFunction> f_ptr(reinterpret_cast<NullaryFunction*>(temp_ptr.get()));

    // copy
    *f_ptr = f;

    // launch
    detail::launch_closure_by_pointer<<<(unsigned int) num_blocks, (unsigned int) block_size>>>(f_ptr.get());

    // free device memory
    thrust::detail::device::cuda::free<0>(f_ptr);
  }
};

} // end detail

template<typename NullaryFunction, typename Size>
  void launch_closure(NullaryFunction f, Size n)
{
  detail::closure_launcher<NullaryFunction>::launch(f, n);
  synchronize_if_enabled("launch_closure");
}

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

