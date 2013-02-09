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


/*! \file for_each.inl
 *  \brief Inline file for for_each.h.
 */

#include <thrust/detail/config.h>

#include <thrust/detail/minmax.h>
#include <thrust/detail/static_assert.h>

#include <thrust/distance.h>
#include <thrust/for_each.h>
#include <thrust/system/cuda/detail/detail/launch_closure.h>
#include <thrust/system/cuda/detail/detail/launch_calculator.h>
#include <thrust/detail/util/blocking.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/function.h>

#include <limits>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace for_each_n_detail
{


template<typename RandomAccessIterator,
         typename Size,
         typename UnaryFunction,
         typename Context>
struct for_each_n_closure
{
  typedef void result_type;
  typedef Context context_type;

  RandomAccessIterator first;
  Size n;
  thrust::detail::device_function<UnaryFunction,void> f;
  Context context;

  for_each_n_closure(RandomAccessIterator first,
                     Size n,
                     UnaryFunction f,
                     Context context = Context())
    : first(first), n(n), f(f), context(context)
  {}

  __device__ __thrust_forceinline__
  result_type operator()(void)
  {
    const Size grid_size = context.block_dimension() * context.grid_dimension();

    Size i = context.linear_index();

    // advance iterator
    first += i;

    while(i < n)
    {
      f(*first);
      i += grid_size;
      first += grid_size;
    }
  }
}; // end for_each_n_closure


template<typename Closure, typename Size>
thrust::tuple<size_t,size_t> configure_launch(Size n)
{
  // calculate launch configuration
  detail::launch_calculator<Closure> calculator;
  
  thrust::tuple<size_t, size_t, size_t> config = calculator.with_variable_block_size();
  size_t max_blocks = thrust::get<0>(config);
  size_t block_size = thrust::get<1>(config);
  size_t num_blocks = thrust::min(max_blocks, thrust::detail::util::divide_ri<size_t>(n, block_size));

  return thrust::make_tuple(num_blocks, block_size);
}


template<typename Size>
bool use_big_closure(Size n, unsigned int little_grid_size)
{
  // use the big closure when n will not fit within an unsigned int
  // or if incrementing an unsigned int by little_grid_size would overflow
  // the counter
  
  Size threshold = std::numeric_limits<unsigned int>::max();

  bool result = (sizeof(Size) > sizeof(unsigned int)) && (n > threshold);

  if(!result)
  {
    // check if we'd overflow the little closure's counter
    unsigned int little_n = static_cast<unsigned int>(n);

    if((little_n - 1u) + little_grid_size < little_n)
    {
      result = true;
    }
  }

  return result;
}


} // end for_each_n_detail


template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename Size,
         typename UnaryFunction>
RandomAccessIterator for_each_n(execution_policy<DerivedPolicy> &,
                                RandomAccessIterator first,
                                Size n,
                                UnaryFunction f)
{
  // we're attempting to launch a kernel, assert we're compiling with nvcc
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to compile your code using nvcc, rather than g++ or cl.exe  X
  // ========================================================================
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<RandomAccessIterator, THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC>::value) );

  if(n <= 0) return first;  // empty range
  
  // create two candidate closures to implement the for_each
  // choose between them based on the whether we can fit n into a smaller integer
  // and whether or not we'll overflow the closure's counter

  typedef detail::blocked_thread_array Context;
  typedef for_each_n_detail::for_each_n_closure<RandomAccessIterator, Size, UnaryFunction, Context>         BigClosure;
  typedef for_each_n_detail::for_each_n_closure<RandomAccessIterator, unsigned int, UnaryFunction, Context> LittleClosure;

  BigClosure    big_closure(first, n, f);
  LittleClosure little_closure(first, static_cast<unsigned int>(n), f);

  thrust::tuple<size_t, size_t> little_config = for_each_n_detail::configure_launch<LittleClosure>(n);

  unsigned int little_grid_size = thrust::get<0>(little_config) * thrust::get<1>(little_config);

  if(for_each_n_detail::use_big_closure(n, little_grid_size))
  {
    // launch the big closure
    thrust::tuple<size_t, size_t> big_config = for_each_n_detail::configure_launch<BigClosure>(n);
    detail::launch_closure(big_closure, thrust::get<0>(big_config), thrust::get<1>(big_config));
  }
  else
  {
    // launch the little closure
    detail::launch_closure(little_closure, thrust::get<0>(little_config), thrust::get<1>(little_config));
  }

  return first + n;
} 


template<typename DerivedPolicy,
         typename InputIterator,
         typename UnaryFunction>
  InputIterator for_each(execution_policy<DerivedPolicy> &exec,
                         InputIterator first,
                         InputIterator last,
                         UnaryFunction f)
{
  return cuda::detail::for_each_n(exec, first, thrust::distance(first,last), f);
} // end for_each()


} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

