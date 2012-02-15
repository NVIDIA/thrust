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


/*! \file unique.h
 *  \brief Entry points for unique backend.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/detail/minimum_space.h>
#include <thrust/detail/backend/generic/unique.h>
#include <thrust/detail/backend/cpp/unique.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace dispatch
{



template<typename ForwardIterator,
         typename BinaryPredicate>
ForwardIterator unique(ForwardIterator first,
                       ForwardIterator last,
                       BinaryPredicate binary_pred,
                       thrust::host_space_tag)
{
    return thrust::detail::backend::cpp::unique(first, last, binary_pred);
}


template<typename ForwardIterator,
         typename BinaryPredicate,
         typename Backend>
ForwardIterator unique(ForwardIterator first,
                       ForwardIterator last,
                       BinaryPredicate binary_pred,
                       Backend)
{
    return thrust::detail::backend::generic::unique(first, last, binary_pred);
}



template<typename InputIterator,
         typename OutputIterator,
         typename BinaryPredicate>
OutputIterator unique_copy(InputIterator first,
                           InputIterator last,
                           OutputIterator output,
                           BinaryPredicate binary_pred,
                           thrust::host_space_tag)
{
    return thrust::detail::backend::cpp::unique_copy(first, last, output, binary_pred);
}


template<typename InputIterator,
         typename OutputIterator,
         typename BinaryPredicate,
         typename Backend>
OutputIterator unique_copy(InputIterator first,
                           InputIterator last,
                           OutputIterator output,
                           BinaryPredicate binary_pred,
                           Backend)
{
    return thrust::detail::backend::generic::unique_copy(first, last, output, binary_pred);
}


template<typename ForwardIterator1,
         typename ForwardIterator2,
         typename BinaryPredicate>
  thrust::pair<ForwardIterator1,ForwardIterator2>
  unique_by_key(ForwardIterator1 keys_first, 
                ForwardIterator1 keys_last,
                ForwardIterator2 values_first,
                BinaryPredicate binary_pred)
{
    return thrust::detail::backend::cpp::unique_by_key(keys_first, keys_last, values_first, binary_pred);
}


template<typename ForwardIterator1,
         typename ForwardIterator2,
         typename BinaryPredicate,
         typename Backend>
  thrust::pair<ForwardIterator1,ForwardIterator2>
  unique_by_key(ForwardIterator1 keys_first, 
                ForwardIterator1 keys_last,
                ForwardIterator2 values_first,
                BinaryPredicate binary_pred,
                Backend)
{
    return thrust::detail::backend::generic::unique_by_key(keys_first, keys_last, values_first, binary_pred);
}



template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename Backend>
  thrust::pair<OutputIterator1,OutputIterator2>
  unique_by_key_copy(InputIterator1 keys_first, 
                     InputIterator1 keys_last,
                     InputIterator2 values_first,
                     OutputIterator1 keys_output,
                     OutputIterator2 values_output,
                     BinaryPredicate binary_pred,
                     Backend)
{
    return thrust::detail::backend::generic::unique_by_key_copy(keys_first, keys_last, values_first, keys_output, values_output, binary_pred);
}



template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate>
  thrust::pair<OutputIterator1,OutputIterator2>
  unique_by_key_copy(InputIterator1 keys_first, 
                     InputIterator1 keys_last,
                     InputIterator2 values_first,
                     OutputIterator1 keys_output,
                     OutputIterator2 values_output,
                     BinaryPredicate binary_pred,
                     thrust::host_space_tag)
{
    return thrust::detail::backend::cpp::unique_by_key_copy(keys_first, keys_last, values_first, keys_output, values_output, binary_pred);
}

} // end namespace dispatch



template<typename ForwardIterator,
         typename BinaryPredicate>
ForwardIterator unique(ForwardIterator first,
                       ForwardIterator last,
                       BinaryPredicate binary_pred)
{
    return thrust::detail::backend::dispatch::unique(first, last, binary_pred,
      typename thrust::iterator_space<ForwardIterator>::type());
}


template<typename InputIterator,
         typename OutputIterator,
         typename BinaryPredicate>
OutputIterator unique_copy(InputIterator first,
                           InputIterator last,
                           OutputIterator output,
                           BinaryPredicate binary_pred)
{
  return thrust::detail::backend::dispatch::unique_copy(first, last, output, binary_pred,
    typename thrust::detail::minimum_space<
      typename thrust::iterator_space<InputIterator>::type,
      typename thrust::iterator_space<OutputIterator>::type
    >::type());
}


template<typename ForwardIterator1,
         typename ForwardIterator2,
         typename BinaryPredicate>
  thrust::pair<ForwardIterator1,ForwardIterator2>
  unique_by_key(ForwardIterator1 keys_first, 
                ForwardIterator1 keys_last,
                ForwardIterator2 values_first,
                BinaryPredicate binary_pred)
{
    return thrust::detail::backend::dispatch::unique_by_key(keys_first, keys_last, values_first, binary_pred,
      typename thrust::detail::minimum_space<
        typename thrust::iterator_space<ForwardIterator1>::type,
        typename thrust::iterator_space<ForwardIterator2>::type
      >::type());
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate>
  thrust::pair<OutputIterator1,OutputIterator2>
  unique_by_key_copy(InputIterator1 keys_first, 
                     InputIterator1 keys_last,
                     InputIterator2 values_first,
                     OutputIterator1 keys_output,
                     OutputIterator2 values_output,
                     BinaryPredicate binary_pred)
{
    return thrust::detail::backend::dispatch::unique_by_key_copy(keys_first, keys_last, values_first, keys_output, values_output, binary_pred,
      typename thrust::detail::minimum_space<
        typename thrust::iterator_space<InputIterator1>::type,
        typename thrust::iterator_space<InputIterator2>::type,
        typename thrust::iterator_space<OutputIterator1>::type,
        typename thrust::iterator_space<OutputIterator2>::type
      >::type());
}


} // end namespace backend
} // end namespace detail
} // end namespace thrust

