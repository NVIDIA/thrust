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

#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/detail/backend_iterator_spaces.h>

#include <thrust/detail/backend/cpp/reduce.h>
#include <thrust/detail/backend/cuda/reduce.h>
#include <thrust/detail/backend/cuda/reduce_by_key.h>
#include <thrust/detail/backend/generic/reduce.h>
#include <thrust/detail/backend/generic/reduce_by_key.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace dispatch
{

template<typename InputIterator, 
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce(InputIterator first,
                    InputIterator last,
                    OutputType init,
                    BinaryFunction binary_op,
                    thrust::host_space_tag)
{
  return thrust::detail::backend::cpp::reduce(first, last, init, binary_op);
}

template<typename InputIterator, 
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce(InputIterator first,
                    InputIterator last,
                    OutputType init,
                    BinaryFunction binary_op,
                    thrust::detail::cuda_device_space_tag)
{
  return thrust::detail::backend::cuda::reduce_n(first, last - first, init, binary_op);
}

template<typename InputIterator, 
         typename OutputType,
         typename BinaryFunction,
         typename Space>
  OutputType reduce(InputIterator first,
                    InputIterator last,
                    OutputType init,
                    BinaryFunction binary_op,
                    Space)
{
  return thrust::detail::backend::generic::reduce_n(first, last - first, init, binary_op);
}

template<typename InputIterator, 
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce(InputIterator first,
                    InputIterator last,
                    OutputType init,
                    BinaryFunction binary_op,
                    thrust::any_space_tag)
{
  return thrust::detail::backend::dispatch::reduce(first, last, init, binary_op,
      thrust::detail::default_device_space_tag());
}



template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction,
         typename Backend>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(InputIterator1 keys_first, 
                     InputIterator1 keys_last,
                     InputIterator2 values_first,
                     OutputIterator1 keys_output,
                     OutputIterator2 values_output,
                     BinaryPredicate binary_pred,
                     BinaryFunction binary_op,
                     Backend)
{
  return thrust::detail::backend::generic::reduce_by_key(keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);
}


template<typename InputIterator1,
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
                     thrust::host_space_tag)
{
  return thrust::detail::backend::cpp::reduce_by_key(keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);
}

template<typename InputIterator1,
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
                     thrust::detail::cuda_device_space_tag)
{
  return thrust::detail::backend::cuda::reduce_by_key(keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);
}

} // end dispatch

// this metafunction passes through a type unless it's any_space_tag,
// in which case it returns default_device_space_tag
template<typename Space>
  struct any_space_to_default_device_space_tag
{
  typedef Space type;
}; // end any_space_to_default_device_space_tag

template<>
  struct any_space_to_default_device_space_tag<thrust::any_space_tag>
{
  typedef thrust::detail::default_device_space_tag type;
}; // end any_space_to_default_device_space_tag


template<typename InputIterator, 
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce(InputIterator first,
                    InputIterator last,
                    OutputType init,
                    BinaryFunction binary_op)
{
  return thrust::detail::backend::dispatch::reduce(first, last, init, binary_op,
      typename thrust::iterator_space<InputIterator>::type());
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
                     BinaryFunction binary_op)
{
  return thrust::detail::backend::dispatch::reduce_by_key(keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op,
      typename thrust::detail::minimum_space<
        typename thrust::iterator_space<InputIterator1>::type,
        typename thrust::iterator_space<InputIterator2>::type,
        typename thrust::iterator_space<OutputIterator1>::type,
        typename thrust::iterator_space<OutputIterator2>::type
      >::type());
}

} // end backend
} // end detail
} // end thrust

