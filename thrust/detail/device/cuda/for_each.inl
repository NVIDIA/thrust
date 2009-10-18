/*
 *  Copyright 2008-2009 NVIDIA Corporation
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

// do not attempt to compile this file with any other compiler
#ifdef __CUDACC__

#include <thrust/detail/device/dereference.h>
#include <thrust/detail/device/cuda/launch_closure.h>

namespace thrust
{

namespace detail
{

namespace device
{

namespace cuda
{

template<typename RandomAccessIterator,
         typename Size,
         typename UnaryFunction>
  struct for_each_n_closure
{
  typedef void result_type;

  RandomAccessIterator first;
  Size n;
  UnaryFunction f;

  for_each_n_closure(RandomAccessIterator first_,
                     Size n_,
                     UnaryFunction f_)
    : first(first_),
      n(n_),
      f(f_)
  {}

  __device__
  result_type operator()(void)
  {
    const Size grid_size = blockDim.x * gridDim.x;

    Size i = blockIdx.x * blockDim.x + threadIdx.x;

    while(i < n)
    {
      f(thrust::detail::device::dereference(first, i));
      i += grid_size;
    }
  }
};


template<typename InputIterator,
         typename UnaryFunction>
void for_each(InputIterator first,
              InputIterator last,
              UnaryFunction f)
{
  if (first >= last) return;  //empty range

  typedef for_each_n_closure<InputIterator, size_t, UnaryFunction> Closure;
  Closure closure(first, last - first, f);

  launch_closure(closure, last - first);
} 


} // end namespace cuda

} // end namespace device

} // end namespace detail

} // end namespace thrust

#endif // __CUDACC__

