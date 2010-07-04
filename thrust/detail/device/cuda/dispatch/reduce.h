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

#include <thrust/pair.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/device/cuda/reduce_n.h>

// XXX remove me when the below function is implemented
#include <stdexcept>


namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{
namespace dispatch
{

template<typename RandomAccessIterator,
         typename SizeType,
         typename OutputType,
         typename BinaryFunction>
  thrust::pair<SizeType,SizeType>
    get_blocked_reduce_n_schedule(RandomAccessIterator first,
                                  SizeType n,
                                  OutputType init,
                                  BinaryFunction binary_op,
                                  thrust::detail::true_type) // use wide reduction
{
  throw std::runtime_error("Unimplemented function: get_blocked_reduce_n_schedule(use_wide_reduction)");
  return thrust::pair<SizeType,SizeType>(0,0);
}

template<typename RandomAccessIterator,
         typename SizeType,
         typename OutputType,
         typename BinaryFunction>
  thrust::pair<SizeType,SizeType>
    get_blocked_reduce_n_schedule(RandomAccessIterator first,
                                  SizeType n,
                                  OutputType init,
                                  BinaryFunction binary_op,
                                  thrust::detail::false_type) // use standard reduction
{
  // standard reduction
  return thrust::detail::device::cuda::detail::get_blocked_reduce_n_schedule(first,n,init,binary_op);
}

} // end dispatch
} // end cuda
} // end device
} // end detail
} // end thrust

