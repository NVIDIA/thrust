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


/*! \file for_each.inl
 *  \brief Inline file for for_each.h.
 */

#include <thrust/detail/config.h>
#include <thrust/for_each.h>
#include <thrust/distance.h>
#include <thrust/system/cuda/detail/bulk.h>
#include <thrust/detail/function.h>
#include <thrust/detail/seq.h>
#include <thrust/system/cuda/detail/execute_on_stream.h>


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


// XXX We could make the kernel simply take first & f as parameters marshalled through async
//     The problem is that confuses -arch=sm_1x compilation -- it causes nvcc to think some
//     local pointers are global instead, which leades to a crash. Wrapping up first & f
//     manually in the functor WARs those problems.
template<typename Iterator, typename Function>
struct for_each_kernel
{
  Iterator first;
  Function f;

  __host__ __device__
  for_each_kernel(Iterator first, Function f)
    : first(first), f(f)
  {}

  __host__ __device__
  void operator()(bulk_::agent<> &self)
  {
    f(first[self.index()]);
  }
};


} // end for_each_n_detail


template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename Size,
         typename UnaryFunction>
__host__ __device__
RandomAccessIterator for_each_n(execution_policy<DerivedPolicy> &exec,
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

  struct workaround
  {
    __host__ __device__
    static RandomAccessIterator parallel_path(execution_policy<DerivedPolicy> &exec, RandomAccessIterator first, Size n, UnaryFunction f)
    {
      typedef for_each_n_detail::for_each_kernel<
        RandomAccessIterator,
        thrust::detail::wrapped_function<UnaryFunction,void>
      > kernel_type;

      kernel_type kernel(first,f);

      bulk_::async(bulk_::par(stream(thrust::detail::derived_cast(exec)), n), kernel, bulk_::root.this_exec);

      return first + n;
    }

    __host__ __device__
    static RandomAccessIterator sequential_path(execution_policy<DerivedPolicy> &, RandomAccessIterator first, Size n, UnaryFunction f)
    {
      return thrust::for_each_n(thrust::seq, first, n, f);
    }
  };

#if __BULK_HAS_CUDART__
  return workaround::parallel_path(exec, first, n, f);
#else
  return workaround::sequential_path(exec, first, n, f);
#endif
} 


template<typename DerivedPolicy,
         typename InputIterator,
         typename UnaryFunction>
__host__ __device__
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

