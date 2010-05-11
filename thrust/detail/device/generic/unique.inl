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


/*! \file unique.inl
 *  \brief Inline file for unique.h.
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <thrust/scan.h>
#include <limits>

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
namespace detail
{

template <typename ValueType, typename TailFlagType>
struct unique_by_key_functor
{
    typedef typename thrust::tuple<ValueType, TailFlagType> result_type;

    __host__ __device__
    result_type operator()(result_type a, result_type b)
    {
        return result_type(thrust::get<1>(b) ? thrust::get<0>(b) : thrust::get<0>(a),
                           thrust::get<1>(a) | thrust::get<1>(b));
    }
};

} // end namespace detail


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
    typedef typename thrust::iterator_space<OutputIterator>::type Space;

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

    return thrust::detail::device::generic::unique_copy_by_key
        (keys.begin(), keys.end(), vals.begin(), keys_first, values_first, binary_pred);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate>
  thrust::pair<OutputIterator1,OutputIterator2>
  unique_copy_by_key(InputIterator1 keys_first, 
                     InputIterator1 keys_last,
                     InputIterator2 values_first,
                     OutputIterator1 keys_output,
                     OutputIterator2 values_output,
                     BinaryPredicate binary_pred)
{
    typedef typename thrust::iterator_traits<InputIterator1>::difference_type difference_type;
    typedef typename thrust::iterator_traits<InputIterator1>::value_type  KeyType;
    typedef typename thrust::iterator_traits<OutputIterator2>::value_type ValueType;
    typedef typename thrust::iterator_space<OutputIterator1>::type Space;
    typedef unsigned int FlagType;

    if (keys_first == keys_last)
        return thrust::make_pair(keys_output, values_output);

    // input size
    difference_type n = keys_last - keys_first;

    InputIterator2 values_last = values_first + n;
    
    // compute head flags
    thrust::detail::raw_buffer<FlagType,Space> head_flags(n);
    thrust::transform(keys_first, keys_last - 1, keys_first + 1, head_flags.begin() + 1, thrust::detail::not2(binary_pred));
    head_flags[0] = 1;

    // compute tail flags
    thrust::detail::raw_buffer<FlagType,Space> tail_flags(n); //COPY INSTEAD OF TRANSFORM
    thrust::transform(keys_first, keys_last - 1, keys_first + 1, tail_flags.begin(), thrust::detail::not2(binary_pred));
    tail_flags[n-1] = 1;

    // scan the values by flag
    thrust::detail::raw_buffer<ValueType,Space> scanned_values(n);
    thrust::detail::raw_buffer<FlagType,Space>  scanned_tail_flags(n);
    
    thrust::detail::device::inclusive_scan
        (thrust::make_zip_iterator(thrust::make_tuple(values_first,           head_flags.begin())),
         thrust::make_zip_iterator(thrust::make_tuple(values_last,            head_flags.end())),
         thrust::make_zip_iterator(thrust::make_tuple(scanned_values.begin(), scanned_tail_flags.begin())),
         detail::unique_by_key_functor<ValueType, FlagType>());

    thrust::exclusive_scan(tail_flags.begin(), tail_flags.end(), scanned_tail_flags.begin());

    // number of unique keys
    FlagType N = scanned_tail_flags[n - 1] + 1;
    
//    std::cout << "head_flags" << std::endl;
//    thrust::copy(head_flags.begin(), head_flags.end(), std::ostream_iterator<FlagType>(std::cout, " "));
//    std::cout << std::endl;
//    
//    std::cout << "tail_flags" << std::endl;
//    thrust::copy(tail_flags.begin(), tail_flags.end(), std::ostream_iterator<FlagType>(std::cout, " "));
//    std::cout << std::endl;
//
//    std::cout << "scanned_tail_flags" << std::endl;
//    thrust::copy(scanned_tail_flags.begin(), scanned_tail_flags.end(), std::ostream_iterator<FlagType>(std::cout, " "));
//    std::cout << std::endl;
//    
//    std::cout << "scanned_values" << std::endl;
//    thrust::copy(scanned_values.begin(), scanned_values.end(), std::ostream_iterator<ValueType>(std::cout, " "));
//    std::cout << std::endl;

    // scatter the keys and accumulated values    
    thrust::scatter_if(keys_first,            keys_last,             scanned_tail_flags.begin(), head_flags.begin(), keys_output);
    thrust::scatter_if(scanned_values.begin(), scanned_values.end(), scanned_tail_flags.begin(), tail_flags.begin(), values_output);

    return thrust::make_pair(keys_output + N, values_output + N); 
}

} // end namespace generic
} // end namespace device
} // end namespace detail
} // end namespace thrust

