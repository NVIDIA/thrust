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


/*! \file reduce.inl
 *  \brief Inline file for reduce.h
 */

#pragma once

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/detail/static_assert.h>

#include <thrust/detail/device/cuda/reduce_n.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{
namespace detail
{
//////////////    
// Functors //
//////////////    
template <typename InputType, typename OutputType, typename BinaryFunction, typename WideType>
  struct wide_unary_op : public thrust::unary_function<WideType,OutputType>
{
    BinaryFunction binary_op;

    __host__ __device__ 
        wide_unary_op(BinaryFunction binary_op) 
            : binary_op(binary_op) {}

    __host__ __device__
        OutputType operator()(WideType x)
        {
            WideType mask = ((WideType) 1 << (8 * sizeof(InputType))) - 1;

            OutputType sum = static_cast<InputType>(x & mask);

            for(unsigned int n = 1; n < sizeof(WideType) / sizeof(InputType); n++)
                sum = binary_op(sum, static_cast<InputType>( (x >> (8 * n * sizeof(InputType))) & mask ) );

            return sum;
        }
};

template<typename InputIterator, 
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce_device(InputIterator first,
                           InputIterator last,
                           OutputType init,
                           BinaryFunction binary_op,
                           thrust::detail::true_type)
{
    // "wide" reduction for small types like char, short, etc.
    typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;
    typedef unsigned int WideType;

    // note: this assumes that InputIterator is a InputType * and can be reinterpret_casted to WideType *
   
    // TODO use simple threshold and ensure alignment of wide_first

    // process first part
    size_t input_type_per_wide_type = sizeof(WideType) / sizeof(InputType);
    size_t n_wide = (last - first) / input_type_per_wide_type;

    const WideType * wide_first = reinterpret_cast<const WideType *>(thrust::raw_pointer_cast(&*first));

    OutputType result = thrust::detail::device::cuda::reduce_n
        (thrust::make_transform_iterator(wide_first, wide_unary_op<InputType,OutputType,BinaryFunction,WideType>(binary_op)),
         n_wide, init, binary_op);

    // process tail
    InputIterator tail_first = first + n_wide * input_type_per_wide_type;
    return thrust::detail::device::cuda::reduce_n(tail_first, last - tail_first, result, binary_op);
}

template<typename InputIterator, 
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce_device(InputIterator first,
                           InputIterator last,
                           OutputType init,
                           BinaryFunction binary_op,
                           thrust::detail::false_type)
{
    // standard reduction
    return thrust::detail::device::cuda::reduce_n(first, last - first, init, binary_op);
}

} // end namespace detail


template<typename InputIterator, 
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce(InputIterator first,
                    InputIterator last,
                    OutputType init,
                    BinaryFunction binary_op)
{
    // we're attempting to launch a kernel, assert we're compiling with nvcc
    // ========================================================================
    // X Note to the user: If you've found this line due to a compiler error, X
    // X you need to compile your code using nvcc, rather than g++ or cl.exe  X
    // ========================================================================
    THRUST_STATIC_ASSERT( (depend_on_instantiation<InputIterator, THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC>::value) );

    typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;

    const bool use_wide_load = thrust::detail::is_pod<InputType>::value 
                                    && thrust::detail::is_trivial_iterator<InputIterator>::value
                                    && (sizeof(InputType) == 1 || sizeof(InputType) == 2);

    // XXX WAR nvcc 3.0 unused variable warning
    (void) use_wide_load;
                                    
    return detail::reduce_device(first, last, init, binary_op, thrust::detail::integral_constant<bool, use_wide_load>());
}

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

