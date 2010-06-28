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


/*! \file internal_functional.inl
 *  \brief Non-public functionals used to implement algorithm internals.
 */

#pragma once

#include <thrust/tuple.h>

namespace thrust
{
namespace detail
{

// unary_negate does not need to know argument_type
template <typename Predicate>
struct unary_negate
{
    Predicate pred;

    __host__ __device__
    explicit unary_negate(const Predicate& pred) : pred(pred) {}

    template <typename T>
        __host__ __device__
        bool operator()(const T& x)
        {
            return !pred(x);
        }
};

// binary_negate does not need to know first_argument_type or second_argument_type
template <typename Predicate>
struct binary_negate
{
    Predicate pred;

    __host__ __device__
    explicit binary_negate(const Predicate& pred) : pred(pred) {}

    template <typename T1, typename T2>
        __host__ __device__
        bool operator()(const T1& x, const T2& y)
        {
            return !pred(x,y);
        }
};

template<typename Predicate>
  __host__ __device__
  thrust::detail::unary_negate<Predicate> not1(const Predicate &pred)
{
    return thrust::detail::unary_negate<Predicate>(pred);
}

template<typename Predicate>
  __host__ __device__
  thrust::detail::binary_negate<Predicate> not2(const Predicate &pred)
{
    return thrust::detail::binary_negate<Predicate>(pred);
}


// convert a predicate to a 0 or 1 integral value
template <typename Predicate, typename IntegralType>
struct predicate_to_integral
{
    Predicate pred;

    __host__ __device__
    explicit predicate_to_integral(const Predicate& pred) : pred(pred) {}

    template <typename T>
        __host__ __device__
        bool operator()(const T& x)
        {
            return pred(x) ? IntegralType(1) : IntegralType(0);
        }
};


// note that detail::equal_to does not force conversion from T2 -> T1 as equal_to does
template <typename T1>
struct equal_to
{
    typedef bool result_type;

    template <typename T2>
        __host__ __device__
        bool operator()(const T1& lhs, const T2& rhs) const
        {
            return lhs == rhs;
        }
};

// note that equal_to_value does not force conversion from T2 -> T1 as equal_to does
template <typename T2>
struct equal_to_value
{
    const T2 rhs;

    equal_to_value(const T2& rhs) : rhs(rhs) {}

    template <typename T1>
        __host__ __device__
        bool operator()(const T1& lhs) const
        {
            return lhs == rhs;
        }
};

template <typename Predicate>
struct tuple_equal_to
{
    typedef bool result_type;

    __host__ __device__
        tuple_equal_to(const Predicate& p) : pred(p) {}

    template<typename Tuple>
        __host__ __device__
        bool operator()(const Tuple& t) const
        { 
            return pred(thrust::get<0>(t), thrust::get<1>(t));
        }

    Predicate pred;
};


template<typename Generator>
  struct generate_functor
{
  typedef void result_type;

  __host__ __device__
  generate_functor(Generator g)
    : gen(g) {}

  template<typename T>
  __host__ __device__
  void operator()(T &x)
  {
    x = gen();
  }

  Generator gen;
};


template<typename ResultType, typename BinaryFunction>
  struct zipped_binary_op
{
  typedef ResultType result_type;

  __host__ __device__
  zipped_binary_op(BinaryFunction binary_op)
    : m_binary_op(binary_op) {}

  template<typename Tuple>
  __host__ __device__
  inline result_type operator()(Tuple t)
  {
    return m_binary_op(thrust::get<0>(t), thrust::get<1>(t));
  }

  BinaryFunction m_binary_op;
};


} // end namespace detail
} // end namespace thrust

