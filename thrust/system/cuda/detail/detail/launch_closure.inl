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

#include <thrust/detail/minmax.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/system/cuda/detail/arch.h>
#include <thrust/system/cuda/detail/synchronize.h>
#include <thrust/system/cuda/detail/detail/launch_calculator.h>
#include <thrust/system/cuda/detail/tag.h>

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
 
  static launch_function_t get_launch_function(void)
  {
    return launch_closure_by_value<Closure>;
  }

  template<typename Size1, typename Size2, typename Size3>
  static void launch(Closure f, Size1 num_blocks, Size2 block_size, Size3 smem_size)
  {
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
    launch_closure_by_value<<<(unsigned int) num_blocks, (unsigned int) block_size, (unsigned int) smem_size>>>(f);
    synchronize_if_enabled("launch_closure_by_value");
#endif // THRUST_DEVICE_COMPILER_NVCC
  }
}; // end closure_launcher_base


template<typename Closure>
  struct closure_launcher_base<Closure,false>
{
  typedef void (*launch_function_t)(const Closure *); 
 
  static launch_function_t get_launch_function(void)
  {
    return launch_closure_by_pointer<Closure>;
  }

  template<typename Size1, typename Size2, typename Size3>
  static void launch(Closure f, Size1 num_blocks, Size2 block_size, Size3 smem_size)
  {
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
    // use temporary storage for the closure
    // XXX use of cuda::tag is too specific here
    thrust::detail::temporary_array<Closure,thrust::cuda::tag> closure_storage(&f, &f + 1);

    // launch
    detail::launch_closure_by_pointer<<<(unsigned int) num_blocks, (unsigned int) block_size, (unsigned int) smem_size>>>((&closure_storage[0]).get());
    synchronize_if_enabled("launch_closure_by_pointer");
#endif // THRUST_DEVICE_COMPILER_NVCC
  }
};


template<typename Closure>
  struct closure_launcher
    : public closure_launcher_base<Closure>
{
  typedef closure_launcher_base<Closure> super_t;
  
  static inline const arch::device_properties_t& device_properties(void)
  {
    return arch::device_properties();
  }
  
  static inline arch::function_attributes_t function_attributes(void)
  {
    return arch::function_attributes(super_t::get_launch_function());
  }

  inline static size_t block_size_with_maximal_occupancy(size_t dynamic_smem_bytes_per_thread = 0)
  {
    return arch::max_blocksize_with_highest_occupancy(super_t::get_launch_function(), dynamic_smem_bytes_per_thread);
  }

  template<typename Size1, typename Size2>
  static size_t num_blocks_with_maximal_occupancy(Size1 n, Size2 block_size, size_t dynamic_smem_bytes_per_block)
  {
    const size_t max_blocks = arch::max_active_blocks(super_t::get_launch_function(), block_size, dynamic_smem_bytes_per_block);

    typedef typename thrust::detail::larger_type<Size1,Size2>::type large_integer;

    // promote both arguments to the larger of Size1 & Size2
    const large_integer result = thrust::min<large_integer>(max_blocks, ( n + (block_size - 1) ) / block_size);

    // due to min above, result is never larger than max_blocks, which is of type size_t
    return static_cast<size_t>(result);
  }

  template<typename Size1, typename Size2, typename Size3>
  static void launch(Closure f, Size1 num_blocks, Size2 block_size, Size3 smem_size)
  {
    super_t::launch(f,num_blocks,block_size,smem_size);
  }
};

template<typename Closure, typename Size>
  void launch_closure(Closure f, Size num_blocks)
{
  launch_calculator<Closure> calculator;
  launch_closure(f, num_blocks, thrust::get<1>(calculator.with_variable_block_size()));
} // end launch_closure()

template<typename Closure, typename Size1, typename Size2>
  void launch_closure(Closure f, Size1 num_blocks, Size2 block_size)
{
  launch_closure(f, num_blocks, block_size, 0u);
} // end launch_closure()

template<typename Closure, typename Size1, typename Size2, typename Size3>
  void launch_closure(Closure f, Size1 num_blocks, Size2 block_size, Size3 smem_size)
{
  closure_launcher<Closure>::launch(f, num_blocks, block_size, smem_size);
} // end launch_closure()

  
template <typename Closure>
arch::function_attributes_t closure_attributes(void)
{
  typedef closure_launcher<Closure> Launcher;

  // cache the result of function_attributes(), because it is slow
  static bool result_exists                 = false;
  static arch::function_attributes_t result = {};

  if(!result_exists)
  {
    result = arch::function_attributes(Launcher::get_launch_function());

    // disallow the compiler to move the write to result_exists
    // before the initialization of result
    __thrust_compiler_fence();

    result_exists = true;
  }

  return result;
}

} // end namespace detail
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

