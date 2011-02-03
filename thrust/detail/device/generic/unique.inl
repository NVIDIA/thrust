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


/*! \file unique.inl
 *  \brief Inline file for unique.h.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>

#include <thrust/detail/raw_buffer.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/internal_functional.h>

#include <thrust/detail/device/scan.h>
#include <thrust/detail/device/copy.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace generic
{

template <typename ForwardIterator,
          typename BinaryPredicate>
ForwardIterator unique(ForwardIterator first,
                       ForwardIterator last,
                       BinaryPredicate binary_pred)
{
    typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;
    typedef typename thrust::iterator_space<ForwardIterator>::type        Space;

    thrust::detail::raw_buffer<InputType,Space> input(first, last);

    return thrust::detail::device::generic::unique_copy(input.begin(), input.end(), first, binary_pred);
}

template <typename InputIterator,
          typename OutputIterator,
          typename BinaryPredicate>
OutputIterator unique_copy(InputIterator first,
                           InputIterator last,
                           OutputIterator output,
                           BinaryPredicate binary_pred)
{
    typedef typename thrust::detail::minimum_space<
      typename thrust::iterator_space<InputIterator>::type,
      typename thrust::iterator_space<OutputIterator>::type
    >::type Space;

    // empty sequence
    if(first == last)
        return output;

    thrust::detail::raw_buffer<int,Space> stencil(thrust::distance(first, last));

    // mark first element in each group
    stencil[0] = 1; 
    thrust::transform(first, last - 1, first + 1, stencil.begin() + 1, thrust::detail::not2(binary_pred)); 

    return thrust::detail::device::copy_if(first, last, stencil.begin(), output, thrust::identity<int>());
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
    typedef typename thrust::iterator_traits<ForwardIterator1>::value_type InputType1;
    typedef typename thrust::iterator_traits<ForwardIterator2>::value_type InputType2;
    typedef typename thrust::iterator_space<ForwardIterator1>::type        Space;

    ForwardIterator2 values_last = values_first + (keys_last - keys_first);

    thrust::detail::raw_buffer<InputType1,Space> keys(keys_first, keys_last);
    thrust::detail::raw_buffer<InputType2,Space> vals(values_first, values_last);

    return thrust::detail::device::generic::unique_by_key_copy
        (keys.begin(), keys.end(), vals.begin(), keys_first, values_first, binary_pred);
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
    typedef typename thrust::iterator_traits<InputIterator1>::difference_type difference_type;
    typedef typename thrust::detail::minimum_space<
      typename thrust::iterator_space<InputIterator1>::type,
      typename thrust::iterator_space<InputIterator2>::type,
      typename thrust::iterator_space<OutputIterator1>::type,
      typename thrust::iterator_space<OutputIterator2>::type
    >::type Space;

    // empty sequence
    if(keys_first == keys_last)
        return thrust::make_pair(keys_output, values_output);

    difference_type n = thrust::distance(keys_first, keys_last);

    thrust::detail::raw_buffer<int,Space> stencil(n);

    // mark first element in each group
    stencil[0] = 1; 
    thrust::transform(keys_first, keys_last - 1, keys_first + 1, stencil.begin() + 1, thrust::detail::not2(binary_pred)); 

    thrust::zip_iterator< thrust::tuple<OutputIterator1, OutputIterator2> > result =
        thrust::detail::device::copy_if(thrust::make_zip_iterator(thrust::make_tuple(keys_first, values_first)),
                                        thrust::make_zip_iterator(thrust::make_tuple(keys_first, values_first)) + n,
                                        stencil.begin(),
                                        thrust::make_zip_iterator(thrust::make_tuple(keys_output, values_output)),
                                        thrust::identity<int>());
    
    difference_type output_size = result - thrust::make_zip_iterator(thrust::make_tuple(keys_output, values_output));
                                    
    return thrust::make_pair(keys_output + output_size, values_output + output_size);
}

} // end namespace generic
} // end namespace device
} // end namespace detail
} // end namespace thrust

