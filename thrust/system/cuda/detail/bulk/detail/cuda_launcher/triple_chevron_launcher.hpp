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

#include <thrust/system/cuda/detail/bulk/detail/config.hpp>
#include <thrust/system/cuda/detail/bulk/detail/alignment.hpp>
#include <thrust/system/cuda/detail/bulk/detail/throw_on_error.hpp>
#include <thrust/system/cuda/detail/bulk/detail/cuda_launcher/parameter_ptr.hpp>

// It's not possible to launch a CUDA kernel unless __BULK_HAS_CUDART__
// is 1, so we'd like to just hide all this code when that macro is 0.
// Unfortunately, we can't actually modulate kernel launches based on that macro
// because that will hide __global__ function template instantiations from critical
// nvcc compilation phases. This means that nvcc won't actually place the kernel in the
// binary and we'll get an undefined __global__ function error at runtime.
// So we allow the user to unconditionally call cuda_launcher.launch() even though it
// will terminate the program at runtime if CUDART is not available.


BULK_NAMESPACE_PREFIX
namespace bulk
{
namespace detail
{


#ifdef __CUDACC__
// if there are multiple versions of Bulk floating around, this may be #defined already
#  ifndef __bulk_launch_bounds__
#    define __bulk_launch_bounds__(num_threads_per_block, num_blocks_per_sm) __launch_bounds__(num_threads_per_block, num_blocks_per_sm)
#  endif
#else
#  ifndef __bulk_launch_bounds__
#    define __bulk_launch_bounds__(num_threads_per_block, num_blocks_per_sm)
#  endif
#endif // __CUDACC__


// triple_chevron_launcher_base is the base class of triple_chevron_launcher
// it primarily serves to choose (statically) which __global__ function is used as the kernel
// sm_20+ devices have 4096 bytes of parameter space
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/#function-parameters
template<unsigned int block_size, typename Function, bool by_value = (sizeof(Function) <= 4096)> struct triple_chevron_launcher_base;


template<unsigned int block_size, typename Function>
__global__
__bulk_launch_bounds__(block_size, 0)
void launch_by_value(Function f)
{
  f();
}


template<unsigned int block_size, typename Function>
struct triple_chevron_launcher_base<block_size,Function,true>
{
  typedef void (*global_function_pointer_t)(Function);

  __host__ __device__
  static global_function_pointer_t global_function_pointer()
  {
    return launch_by_value<block_size,Function>;
  }
};


template<unsigned int block_size, typename Function>
__global__
__bulk_launch_bounds__(block_size, 0)
void launch_by_pointer(const Function *f)
{
  // copy to registers
  Function f_reg = *f;
  f_reg();
}


template<unsigned int block_size, typename Function>
struct triple_chevron_launcher_base<block_size,Function,false>
{
  typedef void (*global_function_pointer_t)(const Function*);

  __host__ __device__
  static global_function_pointer_t global_function_pointer()
  {
    return launch_by_pointer<block_size,Function>;
  }
};


// sm_20+ devices have 4096 bytes of parameter space
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/#function-parameters
template<unsigned int block_size_, typename Function, bool by_value = sizeof(Function) <= 4096>
class triple_chevron_launcher : protected triple_chevron_launcher_base<block_size_, Function>
{
  private:
    typedef triple_chevron_launcher_base<block_size_,Function> super_t;

  public:
    typedef Function task_type;

    inline __host__ __device__
    void launch(unsigned int num_blocks, unsigned int block_size, size_t num_dynamic_smem_bytes, cudaStream_t stream, task_type task)
    {
      struct workaround
      {
        __host__ __device__
        static void supported_path(unsigned int num_blocks, unsigned int block_size, size_t num_dynamic_smem_bytes, cudaStream_t stream, task_type task)
        {
#if __BULK_HAS_CUDART__
#  ifndef __CUDA_ARCH__
          cudaConfigureCall(dim3(num_blocks), dim3(block_size), num_dynamic_smem_bytes, stream);
          cudaSetupArgument(task, 0);
          bulk::detail::throw_on_error(cudaLaunch(super_t::global_function_pointer()), "after cudaLaunch in triple_chevron_launcher::launch()");
#  else
          void *param_buffer = cudaGetParameterBuffer(alignment_of<task_type>::value, sizeof(task_type));
          std::memcpy(param_buffer, &task, sizeof(task_type));
          bulk::detail::throw_on_error(cudaLaunchDevice(reinterpret_cast<void*>(super_t::global_function_pointer()), param_buffer, dim3(num_blocks), dim3(block_size), num_dynamic_smem_bytes, stream),
                                       "after cudaLaunchDevice in triple_chevron_launcher::launch()");
#  endif // __CUDA_ARCH__
#endif // __BULK_HAS_CUDART__
        }

        __host__ __device__
        static void unsupported_path(unsigned int, unsigned int, size_t, cudaStream_t, task_type)
        {
          bulk::detail::terminate_with_message("triple_chevron_launcher::launch(): CUDA kernel launch requires CUDART.");
        }
      };

#if __BULK_HAS_CUDART__
      workaround::supported_path(num_blocks, block_size, num_dynamic_smem_bytes, stream, task);
#else
      workaround::unsupported_path(num_blocks, block_size, num_dynamic_smem_bytes, stream, task);
#endif
    } // end launch()
};


// sm_20+ devices have 4096 bytes of parameter space
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/#function-parameters
// This specialization of triple_chevron_launcher marshals large Functions through
// global memory via parameter_ptr
template<unsigned int block_size_, typename Function>
class triple_chevron_launcher<block_size_,Function,false> : protected triple_chevron_launcher_base<block_size_,Function>
{
  private:
    typedef triple_chevron_launcher_base<block_size_,Function> super_t;

  public:
    typedef Function task_type;

    inline __host__ __device__
    void launch(unsigned int num_blocks, unsigned int block_size, size_t num_dynamic_smem_bytes, cudaStream_t stream, task_type task)
    {
      struct workaround
      {
        __host__ __device__
        static void supported_path(unsigned int num_blocks, unsigned int block_size, size_t num_dynamic_smem_bytes, cudaStream_t stream, task_type task)
        {
          bulk::detail::parameter_ptr<task_type> parm = bulk::detail::make_parameter<task_type>(task);

#if __BULK_HAS_CUDART__
#  ifndef __CUDA_ARCH__
          cudaConfigureCall(dim3(num_blocks), dim3(block_size), num_dynamic_smem_bytes, stream);
          cudaSetupArgument(static_cast<const task_type*>(parm.get()), 0);
          bulk::detail::throw_on_error(cudaLaunch(super_t::global_function_pointer()), "after cudaLaunch in triple_chevron_launcher::launch()");
#  else
          void *param_buffer = cudaGetParameterBuffer(alignment_of<task_type>::value, sizeof(task_type));
          task_type *task_ptr = parm.get();
          std::memcpy(param_buffer, &task_ptr, sizeof(task_type*));
          bulk::detail::throw_on_error(cudaLaunchDevice(reinterpret_cast<void*>(super_t::global_function_pointer()), param_buffer, dim3(num_blocks), dim3(block_size), num_dynamic_smem_bytes, stream),
                                       "after cudaLaunchDevice in triple_chevron_launcher::launch()");
#  endif // __CUDA_ARCH__
#endif // __BULK_HAS_CUDART__
        }

        __host__ __device__
        static void unsupported_path(unsigned int, unsigned int, size_t, cudaStream_t, task_type)
        {
          bulk::detail::terminate_with_message("triple_chevron_launcher::launch(): CUDA kernel launch requires CUDART.");
        }
      };

#if __BULK_HAS_CUDART__
      workaround::supported_path(num_blocks, block_size, num_dynamic_smem_bytes, stream, task);
#else
      workaround::unsupported_path(num_blocks, block_size, num_dynamic_smem_bytes, stream, task);
#endif
    } // end launch()
};


} // end detail
} // end bul
BULK_NAMESPACE_SUFFIX

