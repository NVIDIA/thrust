/*
 *  Copyright 2008-2009 NVIDIA Corporation
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


/*! \file transform.inl
 *  \brief Inline file for transform.h
 */

#pragma once

#include <thrust/detail/device/cuda/vectorize.h>
#include <thrust/detail/device/dereference.h>

namespace thrust
{

namespace detail
{

namespace device
{

namespace detail
{

//////////////    
// Functors //
//////////////    

template <typename InputIterator, typename OutputIterator, typename UnaryFunction>
struct unary_transform_functor
{
    InputIterator first;
    OutputIterator result;
    UnaryFunction unary_op;

    unary_transform_functor(InputIterator _first, OutputIterator _result, UnaryFunction _unary_op)
        : first(_first), result(_result), unary_op(_unary_op) {} 

    template <typename IntegerType>
         __device__
        void operator()(const IntegerType& i)
        { 
            // result[i] = unary_op(first[i]);
            thrust::detail::device::dereference(result, i) = unary_op(thrust::detail::device::dereference(first, i));
        }
}; // end unary_transform_functor


template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
struct binary_transform_functor
{
    InputIterator1 first1;
    InputIterator2 first2;
    OutputIterator result;
    BinaryFunction binary_op;

    binary_transform_functor(InputIterator1 _first1, InputIterator2 _first2, OutputIterator _result, BinaryFunction _binary_op)
        : first1(_first1), first2(_first2), result(_result), binary_op(_binary_op) {} 

    template <typename IntegerType>
        __device__
        void operator()(const IntegerType& i)
        { 
            // result[i] = binary_op(first1[i], first2[i]); 
            thrust::detail::device::dereference(result, i) = binary_op(thrust::detail::device::dereference(first1, i), thrust::detail::device::dereference(first2, i));
        }
}; // end binary_transform_functor


template <typename InputIterator1, typename InputIterator2, typename ForwardIterator, typename UnaryFunction, typename Predicate>
struct unary_transform_if_functor
{
    InputIterator1 first;
    InputIterator2 stencil;
    ForwardIterator result;
    UnaryFunction unary_op;
    Predicate pred;

    unary_transform_if_functor(InputIterator1 _first, InputIterator2 _stencil, ForwardIterator _result, UnaryFunction _unary_op, Predicate _pred)
        : first(_first), stencil(_stencil), result(_result), unary_op(_unary_op), pred(_pred) {} 

    template <typename IntegerType>
        __device__
        void operator()(const IntegerType& i)
        {
            // if(pred(stencil[i]))
            //     result[i] = unary_op(first[i]);
            if(pred(thrust::detail::device::dereference(stencil, i)))
                thrust::detail::device::dereference(result, i) = unary_op(thrust::detail::device::dereference(first, i));
        }
}; // end unary_transform_if_functor


template <typename InputIterator1, typename InputIterator2, typename InputIterator3,
          typename ForwardIterator, typename BinaryFunction, typename Predicate>
struct binary_transform_if_functor
{
    InputIterator1 first1;
    InputIterator2 first2;
    InputIterator3 stencil;
    ForwardIterator result;
    BinaryFunction binary_op;
    Predicate pred;

    binary_transform_if_functor(InputIterator1 _first1, InputIterator2 _first2, InputIterator2 _stencil, 
                                ForwardIterator _result, BinaryFunction _binary_op, Predicate _pred)
        : first1(_first1), first2(_first2), stencil(_stencil), result(_result), binary_op(_binary_op), pred(_pred) {} 
    template <typename IntegerType>
        __device__
        void operator()(const IntegerType& i)
        {
            // if(pred(stencil[i]))
            //     result[i] = binary_op(first1[i], first2[i]);
            if(pred(thrust::detail::device::dereference(stencil, i)))
                thrust::detail::device::dereference(result, i) = binary_op(thrust::detail::device::dereference(first1, i), thrust::detail::device::dereference(first2, i));
        }
}; // end binary_transform_if_functor

} // end namespace detail


template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction>
  OutputIterator transform(InputIterator first, InputIterator last,
                           OutputIterator result,
                           UnaryFunction unary_op)
{
    detail::unary_transform_functor<InputIterator,OutputIterator,UnaryFunction> func(first, result, unary_op);
    thrust::detail::device::cuda::vectorize(last - first, func);
    return result + (last - first); // return the end of the output sequence
} // end transform() 


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryFunction>
  OutputIterator transform(InputIterator1 first1, InputIterator1 last1,
                           InputIterator2 first2,
                           OutputIterator result,
                           BinaryFunction binary_op)
{
    detail::binary_transform_functor<InputIterator1,InputIterator2,OutputIterator,BinaryFunction> func(first1, first2, result, binary_op);
    thrust::detail::device::cuda::vectorize(last1 - first1, func);
    return result + (last1 - first1); // return the end of the output sequence
} // end transform()


template<typename InputIterator1,
         typename InputIterator2,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
  ForwardIterator transform_if(InputIterator1 first, InputIterator1 last,
                               InputIterator2 stencil,
                               ForwardIterator result,
                               UnaryFunction unary_op,
                               Predicate pred)
{
    detail::unary_transform_if_functor<InputIterator1,InputIterator2,ForwardIterator,UnaryFunction,Predicate> func(first, stencil, result, unary_op, pred);
    thrust::detail::device::cuda::vectorize(last - first, func);
    return result + (last - first); // return the end of the output sequence
} // end transform_if()


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename ForwardIterator,
         typename BinaryFunction,
         typename Predicate>
  ForwardIterator transform_if(InputIterator1 first1, InputIterator1 last1,
                               InputIterator2 first2,
                               InputIterator3 stencil,
                               ForwardIterator result,
                               BinaryFunction binary_op,
                               Predicate pred)
{
    detail::binary_transform_if_functor<InputIterator1,InputIterator2,InputIterator3,ForwardIterator,BinaryFunction,Predicate> func(first1, first2, stencil, result, binary_op, pred);
    thrust::detail::device::cuda::vectorize(last1 - first1, func);
    return result + (last1 - first1); // return the end of the output sequence
} // end transform_if()

} // end namespace device

} // end detail

} // end thrust

