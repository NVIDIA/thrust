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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/detail/generic/unique_by_key.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/detail/minimum_system.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/detail/copy_if.h>
#include <thrust/unique.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename System,
         typename ForwardIterator1,
         typename ForwardIterator2>
  thrust::pair<ForwardIterator1,ForwardIterator2>
    unique_by_key(thrust::dispatchable<System> &system,
                  ForwardIterator1 keys_first, 
                  ForwardIterator1 keys_last,
                  ForwardIterator2 values_first)
{
  typedef typename thrust::iterator_traits<ForwardIterator1>::value_type KeyType;
  return thrust::unique_by_key(system, keys_first, keys_last, values_first, thrust::equal_to<KeyType>());
} // end unique_by_key()


template<typename System,
         typename ForwardIterator1,
         typename ForwardIterator2,
         typename BinaryPredicate>
  thrust::pair<ForwardIterator1,ForwardIterator2>
    unique_by_key(thrust::dispatchable<System> &system,
                  ForwardIterator1 keys_first, 
                  ForwardIterator1 keys_last,
                  ForwardIterator2 values_first,
                  BinaryPredicate binary_pred)
{
  typedef typename thrust::iterator_traits<ForwardIterator1>::value_type InputType1;
  typedef typename thrust::iterator_traits<ForwardIterator2>::value_type InputType2;
  
  ForwardIterator2 values_last = values_first + (keys_last - keys_first);
  
  thrust::detail::temporary_array<InputType1,System> keys(system, keys_first, keys_last);
  thrust::detail::temporary_array<InputType2,System> vals(system, values_first, values_last);
  
  return thrust::unique_by_key_copy(system, keys.begin(), keys.end(), vals.begin(), keys_first, values_first, binary_pred);
} // end unique_by_key()


template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
    unique_by_key_copy(thrust::dispatchable<System> &system,
                       InputIterator1 keys_first, 
                       InputIterator1 keys_last,
                       InputIterator2 values_first,
                       OutputIterator1 keys_output,
                       OutputIterator2 values_output)
{
  typedef typename thrust::iterator_traits<InputIterator1>::value_type KeyType;
  return thrust::unique_by_key_copy(system, keys_first, keys_last, values_first, keys_output, values_output, thrust::equal_to<KeyType>());
} // end unique_by_key_copy()


template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate>
  thrust::pair<OutputIterator1,OutputIterator2>
    unique_by_key_copy(thrust::dispatchable<System> &system,
                       InputIterator1 keys_first, 
                       InputIterator1 keys_last,
                       InputIterator2 values_first,
                       OutputIterator1 keys_output,
                       OutputIterator2 values_output,
                       BinaryPredicate binary_pred)
{
  typedef typename thrust::iterator_traits<InputIterator1>::difference_type difference_type;
  
  // empty sequence
  if(keys_first == keys_last)
    return thrust::make_pair(keys_output, values_output);
  
  difference_type n = thrust::distance(keys_first, keys_last);
  
  thrust::detail::temporary_array<int,System> stencil(system,n);
  
  // mark first element in each group
  stencil[0] = 1; 
  thrust::transform(system, keys_first, keys_last - 1, keys_first + 1, stencil.begin() + 1, thrust::detail::not2(binary_pred)); 
  
  thrust::zip_iterator< thrust::tuple<OutputIterator1, OutputIterator2> > result =
    thrust::copy_if(system,
                    thrust::make_zip_iterator(thrust::make_tuple(keys_first, values_first)),
                    thrust::make_zip_iterator(thrust::make_tuple(keys_first, values_first)) + n,
                    stencil.begin(),
                    thrust::make_zip_iterator(thrust::make_tuple(keys_output, values_output)),
                    thrust::identity<int>());
  
  difference_type output_size = result - thrust::make_zip_iterator(thrust::make_tuple(keys_output, values_output));
                                  
  return thrust::make_pair(keys_output + output_size, values_output + output_size);
} // end unique_by_key_copy()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace thrust

