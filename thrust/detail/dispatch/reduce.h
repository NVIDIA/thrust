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


/*! \file transform_reduce.h
 *  \brief Dispatch layer for transform_reduce.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/host/reduce.h>
#include <thrust/detail/device/reduce.h>

namespace thrust
{

namespace detail
{

namespace dispatch
{

////////////////
// Host Paths //
////////////////
template<typename InputIterator, 
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce(InputIterator first,
                    InputIterator last,
                    OutputType init,
                    BinaryFunction binary_op,
                    thrust::host_space_tag)
{
    return thrust::detail::host::reduce(first, last, init, binary_op);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate,
          typename BinaryFunction>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(InputIterator1 keys_first, 
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred,
                BinaryFunction binary_op,
                thrust::host_space_tag,
                thrust::host_space_tag,
                thrust::host_space_tag,
                thrust::host_space_tag)
{
    return thrust::detail::host::reduce_by_key(keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);
}

//////////////////
// Device Paths //
//////////////////
template<typename InputIterator, 
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce(InputIterator first,
                    InputIterator last,
                    OutputType init,
                    BinaryFunction binary_op,
                    thrust::device_space_tag)
{
    return thrust::detail::device::reduce(first, last, init, binary_op);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate,
          typename BinaryFunction>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(InputIterator1 keys_first, 
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred,
                BinaryFunction binary_op,
                thrust::device_space_tag,
                thrust::device_space_tag,
                thrust::device_space_tag,
                thrust::device_space_tag)
{
    return thrust::detail::device::reduce_by_key(keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);
}

//////////////
// Any Path //
//////////////
template<typename InputIterator, 
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce(InputIterator first,
                    InputIterator last,
                    OutputType init,
                    BinaryFunction binary_op,
                    thrust::any_space_tag)
{
    // default to device path
    return thrust::detail::dispatch::reduce(first, last, init, binary_op, thrust::device_space_tag());
}

} // end namespace dispatch
} // end namespace detail
} // end namespace thrust

