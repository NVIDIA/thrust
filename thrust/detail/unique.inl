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


/*! \file unique.inl
 *  \brief Inline file for unique.h.
 */

#include <thrust/unique.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/backend/unique.h>

namespace thrust
{
    
template <typename ForwardIterator>
ForwardIterator unique(ForwardIterator first,
                       ForwardIterator last)
{
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;
  return thrust::unique(first, last, thrust::equal_to<InputType>());
} // end unique()


template <typename ForwardIterator,
          typename BinaryPredicate>
ForwardIterator unique(ForwardIterator first,
                       ForwardIterator last,
                       BinaryPredicate binary_pred)
{
  return detail::backend::unique(first, last, binary_pred);
} // end unique()


template <typename InputIterator,
          typename OutputIterator>
OutputIterator unique_copy(InputIterator first,
                           InputIterator last,
                           OutputIterator output)
{
  typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;
  return thrust::unique_copy(first, last, output, thrust::equal_to<InputType>());
}


template <typename InputIterator,
          typename OutputIterator,
          typename BinaryPredicate>
OutputIterator unique_copy(InputIterator first,
                           InputIterator last,
                           OutputIterator output,
                           BinaryPredicate binary_pred)
{
  return detail::backend::unique_copy(first, last, output, binary_pred);
}

template <typename ForwardIterator1,
          typename ForwardIterator2>
  thrust::pair<ForwardIterator1,ForwardIterator2>
  unique_by_key(ForwardIterator1 keys_first, 
                ForwardIterator1 keys_last,
                ForwardIterator2 values_first)
{
  typedef typename thrust::iterator_traits<ForwardIterator1>::value_type KeyType;
  return thrust::unique_by_key(keys_first, keys_last, values_first, thrust::equal_to<KeyType>());
}

template <typename ForwardIterator1,
          typename ForwardIterator2,
          typename BinaryPredicate>
  thrust::pair<ForwardIterator1,ForwardIterator2>
  unique_by_key(ForwardIterator1 keys_first, 
                ForwardIterator1 keys_last,
                ForwardIterator2 values_first,
                BinaryPredicate binary_pred)
{
  return detail::backend::unique_by_key(keys_first, keys_last, values_first, binary_pred);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
  unique_by_key_copy(InputIterator1 keys_first, 
                     InputIterator1 keys_last,
                     InputIterator2 values_first,
                     OutputIterator1 keys_output,
                     OutputIterator2 values_output)
{
  typedef typename thrust::iterator_traits<InputIterator1>::value_type KeyType;
  return thrust::unique_by_key_copy
      (keys_first, keys_last, values_first, keys_output, values_output, thrust::equal_to<KeyType>());
}

template <typename InputIterator1,
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
  return detail::backend::unique_by_key_copy(keys_first, keys_last, values_first, keys_output, values_output, binary_pred);
}

} // end namespace thrust

