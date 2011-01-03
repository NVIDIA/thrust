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

/*! \file copy.h
 *  \brief CUDA implementation of device-to-device copy,
 *         based on Gregory Diamos' memcpy code.
 */

#pragma once

#include <thrust/detail/type_traits.h>
#include <thrust/detail/device/dereference.h>

#include <thrust/detail/dispatch/is_trivial_copy.h>
#include <thrust/detail/device/cuda/detail/trivial_copy.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{
namespace block
{
namespace detail
{
namespace dispatch
{

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  __forceinline__ __device__
  RandomAccessIterator2 copy(RandomAccessIterator1 first,
                             RandomAccessIterator1 last,
                             RandomAccessIterator2 result,
                             thrust::detail::true_type is_trivial_copy)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type T;

  // XXX these aren't working at the moment
  //const T *src = thrust::raw_pointer_cast(&*first);
  //      T *dst = thrust::raw_pointer_cast(&*result);
  const T *src = &dereference(first);
        T *dst = &dereference(result);

  size_t n = (last - first);
  cuda::detail::trivial_copy<cuda::detail::trivial_copy_block>(dst, src, n * sizeof(T));
  return result + n;
} // end copy()

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  __forceinline__ __device__
  RandomAccessIterator2 copy(RandomAccessIterator1 first,
                             RandomAccessIterator1 last,
                             RandomAccessIterator2 result,
                             thrust::detail::false_type is_trivial_copy)
{
  RandomAccessIterator2 end_of_output = result + (last - first);
  
  // advance iterators
  first  += threadIdx.x;
  result += threadIdx.x;

  for(;
      first < last;
      first += blockDim.x,
      result += blockDim.x)
  {
    dereference(result) = dereference(first);
  } // end for

  return end_of_output;
} // end copy()

} // end namespace dispatch
} // end namespace detail

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  __forceinline__ __device__
  RandomAccessIterator2 copy(RandomAccessIterator1 first,
                             RandomAccessIterator1 last,
                             RandomAccessIterator2 result)
{
  return detail::dispatch::copy(first, last, result,
#if __CUDA_ARCH__ < 200
      // does not work reliably on pre-Fermi due to "Warning: ... assuming global memory space" issues
      false_type()
#else
      typename thrust::detail::dispatch::is_trivial_copy<RandomAccessIterator1,RandomAccessIterator2>::type()
#endif
      );
} // end copy()

} // end namespace block
} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

