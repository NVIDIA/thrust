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

#include <thrust/detail/minmax.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/system/cuda/detail/runtime_introspection.h>
#include <thrust/system/cuda/detail/synchronize.h>
#include <thrust/system/cuda/detail/detail/launch_calculator.h>
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/system/cuda/detail/execute_on_stream.h>
#include <thrust/system/cuda/detail/detail/alignment.h>
#include <thrust/system/cuda/detail/bulk.h>

namespace thrust
{
namespace detail
{

// XXX WAR circular inclusion problems with this forward declaration
template<typename, typename> class temporary_array;

} // end detail

namespace system
{
namespace cuda
{
namespace detail
{
namespace detail
{

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
template<typename Closure>
__global__ __launch_bounds__(Closure::context_type::ThreadsPerBlock::value, Closure::context_type::BlocksPerMultiprocessor::value)
void launch_closure_by_value(Closure f)
{
  f();
}

template<typename Closure>
__global__ __launch_bounds__(Closure::context_type::ThreadsPerBlock::value, Closure::context_type::BlocksPerMultiprocessor::value)
void launch_closure_by_pointer(const Closure *f)
{
  // copy to registers
  Closure f_reg = *f;
  f_reg();
}
#else
template<typename Closure>
void launch_closure_by_value(Closure) {}

template<typename Closure>
void launch_closure_by_pointer(const Closure *) {}

#endif // THRUST_DEVICE_COMPILER_NVCC

template<typename Closure,
         bool launch_by_value = sizeof(Closure) <= 256>
  struct closure_launcher_base
{
  typedef void (*launch_function_t)(Closure); 
 
  __host__ __device__
  static launch_function_t get_launch_function()
  {
    return launch_closure_by_value<Closure>;
  }

  template<typename DerivedPolicy, typename Size1, typename Size2, typename Size3>
  __host__ __device__
  static void launch(execution_policy<DerivedPolicy> &exec, Closure f, Size1 num_blocks, Size2 block_size, Size3 smem_size)
  {
    // this ensures that the kernel gets instantiated identically for all values of __CUDA_ARCH__
    launch_function_t kernel = get_launch_function();

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
#if __BULK_HAS_CUDART__
    if(num_blocks > 0)
    {
#ifndef __CUDA_ARCH__
      kernel<<<(unsigned int) num_blocks, (unsigned int) block_size, (unsigned int) smem_size, stream(thrust::detail::derived_cast(exec))>>>(f);
#else
      // XXX we can't pass parameters with constructors to kernels launched through the triple chevrons in __device__ code
      //     use cudaLaunchDevice directly
      void *param_buffer = cudaGetParameterBuffer(alignment_of<Closure>::value, sizeof(Closure));
      std::memcpy(param_buffer, &f, sizeof(Closure));
      cudaLaunchDevice(reinterpret_cast<void*>(kernel), param_buffer, dim3(num_blocks), dim3(block_size), smem_size, stream(thrust::detail::derived_cast(exec)));
#endif // __CUDA_ARCH__
      synchronize_if_enabled("launch_closure_by_value");
    }
#endif // __BULK_HAS_CUDART__
#endif // THRUST_DEVICE_COMPILER_NVCC
  }
}; // end closure_launcher_base


template<typename Closure>
  struct closure_launcher_base<Closure,false>
{
  typedef void (*launch_function_t)(const Closure *); 
 
  __host__ __device__
  static launch_function_t get_launch_function(void)
  {
    return launch_closure_by_pointer<Closure>;
  }

  template<typename DerivedPolicy, typename Size1, typename Size2, typename Size3>
  __host__ __device__
  static void launch(execution_policy<DerivedPolicy> &exec, Closure f, Size1 num_blocks, Size2 block_size, Size3 smem_size)
  {
    // this ensures that the kernel gets instantiated identically for all values of __CUDA_ARCH__
    launch_function_t kernel = get_launch_function();

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
#if __BULK_HAS_CUDART__
    if(num_blocks > 0)
    {
      // use temporary storage for the closure
      thrust::host_system_tag host_tag;
      thrust::detail::temporary_array<Closure,DerivedPolicy> closure_storage(exec, host_tag, &f, &f + 1);

      // launch
      kernel<<<(unsigned int) num_blocks, (unsigned int) block_size, (unsigned int) smem_size, stream(thrust::detail::derived_cast(exec))>>>((&closure_storage[0]).get());
      synchronize_if_enabled("launch_closure_by_pointer");
    }
#endif // __BULK_HAS_CUDART__
#endif // THRUST_DEVICE_COMPILER_NVCC
  }
};


template<typename Closure>
  struct closure_launcher
    : public closure_launcher_base<Closure>
{
  typedef closure_launcher_base<Closure> super_t;
  
  __host__ __device__
  static inline const device_properties_t& device_properties(void)
  {
    return device_properties();
  }
  
  __host__ __device__
  static inline function_attributes_t function_attributes(void)
  {
    return thrust::system::cuda::detail::function_attributes(super_t::get_launch_function());
  }

  template<typename DerivedPolicy, typename Size1, typename Size2, typename Size3>
  __host__ __device__
  static void launch(execution_policy<DerivedPolicy> &exec, Closure f, Size1 num_blocks, Size2 block_size, Size3 smem_size)
  {
    super_t::launch(exec,f,num_blocks,block_size,smem_size);
  }
};

template<typename DerivedPolicy, typename Closure, typename Size>
__host__ __device__
void launch_closure(execution_policy<DerivedPolicy> &exec, Closure f, Size num_blocks)
{
  launch_calculator<Closure> calculator;
  launch_closure(exec, f, num_blocks, thrust::get<1>(calculator.with_variable_block_size()));
} // end launch_closure()

template<typename DerivedPolicy, typename Closure, typename Size1, typename Size2>
__host__ __device__
void launch_closure(execution_policy<DerivedPolicy> &exec, Closure f, Size1 num_blocks, Size2 block_size)
{
  launch_closure(exec, f, num_blocks, block_size, 0u);
} // end launch_closure()

template<typename DerivedPolicy, typename Closure, typename Size1, typename Size2, typename Size3>
__host__ __device__
void launch_closure(execution_policy<DerivedPolicy> &exec, Closure f, Size1 num_blocks, Size2 block_size, Size3 smem_size)
{
  closure_launcher<Closure>::launch(exec, f, num_blocks, block_size, smem_size);
} // end launch_closure()


namespace closure_attributes_detail
{


template<typename Closure>
inline __host__ __device__
function_attributes_t uncached_closure_attributes()
{
  typedef closure_launcher<Closure> Launcher;
  return thrust::system::cuda::detail::function_attributes(Launcher::get_launch_function());
}


template<typename Closure>
function_attributes_t cached_closure_attributes()
{
  // cache the result of function_attributes(), because it is slow
  // only cache the first few devices
  static const int max_num_devices                                  = 16;

  static bool attributes_exist[max_num_devices]                     = {0};
  static function_attributes_t function_attributes[max_num_devices] = {};

  // XXX device_id ought to be an argument to this function
  int device_id = current_device();

  if(device_id >= max_num_devices)
  {
    return uncached_closure_attributes<Closure>();
  }

  if(!attributes_exist[device_id])
  {
    function_attributes[device_id] = uncached_closure_attributes<Closure>();

    // disallow the compiler to move the write to attributes_exist[device_id]
    // before the initialization of function_attributes[device_id]
    __thrust_compiler_fence();

    attributes_exist[device_id] = true;
  }

  return function_attributes[device_id];
}


} // end closure_attributes_detail

  
template<typename Closure>
__host__ __device__
function_attributes_t closure_attributes()
{
#ifndef __CUDA_ARCH__
  return closure_attributes_detail::cached_closure_attributes<Closure>();
#else
  return closure_attributes_detail::uncached_closure_attributes<Closure>();
#endif
}

} // end namespace detail
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

