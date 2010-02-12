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

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <thrust/detail/device/reduce.h>

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

// given a tuple (x,y) return binary_op2(x,y)
template <typename OutputType, typename BinaryFunction2>
struct inner_product_functor
{
    typedef OutputType result_type;
    BinaryFunction2 binary_op2;

    inner_product_functor(BinaryFunction2 _binary_op2) 
        : binary_op2(_binary_op2) {}

    template <typename TupleType>
        __host__ __device__
        OutputType operator()(TupleType t)
        { 
            return binary_op2(thrust::get<0>(t), thrust::get<1>(t));
        }
}; // end inner_product_functor

} // end namespace detail

template <typename InputIterator1, typename InputIterator2, typename OutputType,
          typename BinaryFunction1, typename BinaryFunction2>
    OutputType
    inner_product(InputIterator1 first1, InputIterator1 last1,
                  InputIterator2 first2, OutputType init, 
                  BinaryFunction1 binary_op1, BinaryFunction2 binary_op2)
{
    detail::inner_product_functor<OutputType, BinaryFunction2> func(binary_op2);

    InputIterator2 last2 = first2 + (last1 - first1);

    return thrust::detail::device::reduce(
            thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(first1, first2)), func),
            thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(last1, last2)), func),
            init,
            binary_op1);
}

} // end namespace generic
} // end namespace device
} // end namespace detail
} // end namespace thrust

