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


/*! \file internal_functional.inl
 *  \brief Non-public functionals used to implement algorithm internals.
 */

#pragma once

#include <thrust/tuple.h>
#include <thrust/iterator/iterator_traits.h>
#include <memory> // for ::new

namespace thrust
{
namespace detail
{

// unary_negate does not need to know argument_type
template <typename Predicate>
struct unary_negate
{
    typedef bool result_type;

    Predicate pred;

    __host__ __device__
    explicit unary_negate(const Predicate& pred) : pred(pred) {}

    template <typename T>
    __host__ __device__
    bool operator()(const T& x)
    {
        return !bool(pred(x));
    }
};

// binary_negate does not need to know first_argument_type or second_argument_type
template <typename Predicate>
struct binary_negate
{
    typedef bool result_type;

    Predicate pred;

    __host__ __device__
    explicit binary_negate(const Predicate& pred) : pred(pred) {}

    template <typename T1, typename T2>
        __host__ __device__
        bool operator()(const T1& x, const T2& y)
        {
            return !bool(pred(x,y));
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
    T2 rhs;

    equal_to_value(const T2& rhs) : rhs(rhs) {}

    template <typename T1>
        __host__ __device__
        bool operator()(const T1& lhs) const
        {
            return lhs == rhs;
        }
};

template <typename Predicate>
struct tuple_binary_predicate
{
    typedef bool result_type;

    __host__ __device__
        tuple_binary_predicate(const Predicate& p) : pred(p) {}

    template<typename Tuple>
        __host__ __device__
        bool operator()(const Tuple& t) const
        { 
            return pred(thrust::get<0>(t), thrust::get<1>(t));
        }

    Predicate pred;
};

template <typename Predicate>
struct tuple_not_binary_predicate
{
    typedef bool result_type;

    __host__ __device__
        tuple_not_binary_predicate(const Predicate& p) : pred(p) {}

    template<typename Tuple>
        __host__ __device__
        bool operator()(const Tuple& t) const
        { 
            return !pred(thrust::get<0>(t), thrust::get<1>(t));
        }

    Predicate pred;
};

template<typename Generator>
  struct host_generate_functor
{
  typedef void result_type;

  __host__ __device__
  host_generate_functor(Generator g)
    : gen(g) {}

  // operator() does not take an lvalue reference because some iterators
  // produce temporary proxy references when dereferenced. for example,
  // consider the temporary tuple of references produced by zip_iterator.
  // such temporaries cannot bind to an lvalue reference.
  //
  // to WAR this, accept a const reference (which is bindable to a temporary),
  // and const_cast in the implementation.
  //
  // XXX change to an rvalue reference upon c++0x (which either a named variable
  //     or temporary can bind to)
  template<typename T>
  __host__
  void operator()(const T &x)
  {
    // we have to be naughty and const_cast this to get it to work
    T &lvalue = const_cast<T&>(x);

    // this assigns correctly whether x is a true reference or proxy
    lvalue = gen();
  }

  Generator gen;
};

template<typename Generator>
  struct device_generate_functor
{
  typedef void result_type;

  __host__ __device__
  device_generate_functor(Generator g)
    : gen(g) {}

  // operator() does not take an lvalue reference because some iterators
  // produce temporary proxy references when dereferenced. for example,
  // consider the temporary tuple of references produced by zip_iterator.
  // such temporaries cannot bind to an lvalue reference.
  //
  // to WAR this, accept a const reference (which is bindable to a temporary),
  // and const_cast in the implementation.
  //
  // XXX change to an rvalue reference upon c++0x (which either a named variable
  //     or temporary can bind to)
  template<typename T>
  __host__ __device__
  void operator()(const T &x)
  {
    // we have to be naughty and const_cast this to get it to work
    T &lvalue = const_cast<T&>(x);

    // this assigns correctly whether x is a true reference or proxy
    lvalue = gen();
  }

  Generator gen;
};

template<typename Space, typename Generator>
  struct generate_functor
    : thrust::detail::eval_if<
        thrust::detail::is_convertible<Space, thrust::host_space_tag>::value,
        thrust::detail::identity_<host_generate_functor<Generator> >,
        thrust::detail::identity_<device_generate_functor<Generator> >
      >
{};


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

template<typename UnaryFunction>
  struct host_unary_transform_functor
{
  typedef void result_type;

  UnaryFunction f;

  host_unary_transform_functor(UnaryFunction f_)
    :f(f_) {}

  template<typename Tuple>
  __host__
  inline result_type operator()(Tuple t)
  {
    thrust::get<1>(t) = f(thrust::get<0>(t));
  }
};

template<typename UnaryFunction>
  struct device_unary_transform_functor
{
  typedef void result_type;

  UnaryFunction f;

  device_unary_transform_functor(UnaryFunction f_)
    :f(f_) {}

  // add __host__ to allow the omp backend compile with nvcc
  template<typename Tuple>
  __host__ __device__
  inline result_type operator()(Tuple t)
  {
    thrust::get<1>(t) = f(thrust::get<0>(t));
  }
};


template<typename Space, typename UnaryFunction>
  struct unary_transform_functor
    : thrust::detail::eval_if<
        thrust::detail::is_convertible<Space, thrust::host_space_tag>::value,
        thrust::detail::identity_<host_unary_transform_functor<UnaryFunction> >,
        thrust::detail::identity_<device_unary_transform_functor<UnaryFunction> >
      >
{};


template <typename BinaryFunction>
struct host_binary_transform_functor
{
  BinaryFunction f;

  host_binary_transform_functor(BinaryFunction f_)
    :f(f_)
  {}

  template <typename Tuple>
  __host__
  void operator()(Tuple t)
  { 
    thrust::get<2>(t) = f(thrust::get<0>(t), thrust::get<1>(t));
  }
}; // end binary_transform_functor


template <typename BinaryFunction>
struct device_binary_transform_functor
{
  BinaryFunction f;

  device_binary_transform_functor(BinaryFunction f_)
    :f(f_)
  {}

  // add __host__ to allow the omp backend compile with nvcc
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  { 
    thrust::get<2>(t) = f(thrust::get<0>(t), thrust::get<1>(t));
  }
}; // end binary_transform_functor


template<typename Space, typename BinaryFunction>
  struct binary_transform_functor
    : thrust::detail::eval_if<
        thrust::detail::is_convertible<Space, thrust::host_space_tag>::value,
        thrust::detail::identity_<host_binary_transform_functor<BinaryFunction> >,
        thrust::detail::identity_<device_binary_transform_functor<BinaryFunction> >
      >
{};


template <typename UnaryFunction, typename Predicate>
struct host_unary_transform_if_functor
{
  UnaryFunction unary_op;
  Predicate pred;

  host_unary_transform_if_functor(UnaryFunction unary_op_, Predicate pred_)
    : unary_op(unary_op_), pred(pred_) {}

  template<typename Tuple>
  __host__
  void operator()(Tuple t)
  {
    if(pred(thrust::get<0>(t)))
    {
      thrust::get<1>(t) = unary_op(thrust::get<0>(t));
    }
  }
}; // end host_unary_transform_if_functor


template <typename UnaryFunction, typename Predicate>
struct device_unary_transform_if_functor
{
  UnaryFunction unary_op;
  Predicate pred;

  device_unary_transform_if_functor(UnaryFunction unary_op_, Predicate pred_)
    : unary_op(unary_op_), pred(pred_) {}

  template<typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    if(pred(thrust::get<0>(t)))
    {
      thrust::get<1>(t) = unary_op(thrust::get<0>(t));
    }
  }
}; // end device_unary_transform_if_functor


template<typename Space, typename UnaryFunction, typename Predicate>
  struct unary_transform_if_functor
    : thrust::detail::eval_if<
        thrust::detail::is_convertible<Space, thrust::host_space_tag>::value,
        thrust::detail::identity_<host_unary_transform_if_functor<UnaryFunction,Predicate> >,
        thrust::detail::identity_<device_unary_transform_if_functor<UnaryFunction,Predicate> >
      >
{};


template <typename UnaryFunction, typename Predicate>
struct host_unary_transform_if_with_stencil_functor
{
  UnaryFunction unary_op;
  Predicate pred;
  
  host_unary_transform_if_with_stencil_functor(UnaryFunction _unary_op, Predicate _pred)
    : unary_op(_unary_op), pred(_pred) {} 
  
  template <typename Tuple>
  __host__
  void operator()(Tuple t)
  {
    if(pred(thrust::get<1>(t)))
      thrust::get<2>(t) = unary_op(thrust::get<0>(t));
  }
}; // end host_unary_transform_if_with_stencil_functor


template <typename UnaryFunction, typename Predicate>
struct device_unary_transform_if_with_stencil_functor
{
  UnaryFunction unary_op;
  Predicate pred;
  
  device_unary_transform_if_with_stencil_functor(UnaryFunction _unary_op, Predicate _pred)
    : unary_op(_unary_op), pred(_pred) {} 
  
  // add __host__ to allow the omp backend compile with nvcc
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    if(pred(thrust::get<1>(t)))
      thrust::get<2>(t) = unary_op(thrust::get<0>(t));
  }
}; // end device_unary_transform_if_with_stencil_functor


template<typename Space, typename UnaryFunction, typename Predicate>
  struct unary_transform_if_with_stencil_functor
    : thrust::detail::eval_if<
        thrust::detail::is_convertible<Space, thrust::host_space_tag>::value,
        thrust::detail::identity_<host_unary_transform_if_with_stencil_functor<UnaryFunction,Predicate> >,
        thrust::detail::identity_<device_unary_transform_if_with_stencil_functor<UnaryFunction,Predicate> >
      >
{};


template <typename BinaryFunction, typename Predicate>
struct host_binary_transform_if_functor
{
  BinaryFunction binary_op;
  Predicate pred;

  host_binary_transform_if_functor(BinaryFunction _binary_op, Predicate _pred)
    : binary_op(_binary_op), pred(_pred) {} 

  template <typename Tuple>
  __host__
  void operator()(Tuple t)
  {
    if(pred(thrust::get<2>(t)))
      thrust::get<3>(t) = binary_op(thrust::get<0>(t), thrust::get<1>(t));
  }
}; // end host_binary_transform_if_functor


template <typename BinaryFunction, typename Predicate>
struct device_binary_transform_if_functor
{
  BinaryFunction binary_op;
  Predicate pred;

  device_binary_transform_if_functor(BinaryFunction _binary_op, Predicate _pred)
    : binary_op(_binary_op), pred(_pred) {} 

  // add __host__ to allow the omp backend compile with nvcc
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    if(pred(thrust::get<2>(t)))
      thrust::get<3>(t) = binary_op(thrust::get<0>(t), thrust::get<1>(t));
  }
}; // end device_binary_transform_if_functor


template<typename Space, typename BinaryFunction, typename Predicate>
  struct binary_transform_if_functor
    : thrust::detail::eval_if<
        thrust::detail::is_convertible<Space, thrust::host_space_tag>::value,
        thrust::detail::identity_<host_binary_transform_if_functor<BinaryFunction,Predicate> >,
        thrust::detail::identity_<device_binary_transform_if_functor<BinaryFunction,Predicate> >
      >
{};


template<typename T>
  struct host_destroy_functor
{
  __host__
  void operator()(T &x) const
  {
    x.~T();
  } // end operator()()
}; // end host_destroy_functor


template<typename T>
  struct device_destroy_functor
{
  // add __host__ to allow the omp backend to compile with nvcc
  __host__ __device__
  void operator()(T &x) const
  {
    x.~T();
  } // end operator()()
}; // end device_destroy_functor


template<typename Space, typename T>
  struct destroy_functor
    : thrust::detail::eval_if<
        thrust::detail::is_convertible<Space, thrust::host_space_tag>::value,
        thrust::detail::identity_<host_destroy_functor<T> >,
        thrust::detail::identity_<device_destroy_functor<T> >
      >
{};


template <typename T>
struct fill_functor
{
  const T exemplar;

  fill_functor(const T& _exemplar) 
    : exemplar(_exemplar) {}

  __host__ __device__
  T operator()(void) const
  { 
    return exemplar;
  }
};


template<typename T>
  struct uninitialized_fill_functor
{
  T exemplar;

  uninitialized_fill_functor(T x):exemplar(x){}

  __host__ __device__
  void operator()(T &x)
  {
    ::new(static_cast<void*>(&x)) T(exemplar);
  } // end operator()()
}; // end uninitialized_fill_functor


// this predicate tests two two-element tuples
// we first use a Compare for the first element
// if the first elements are equivalent, we use
// < for the second elements
template<typename Compare>
  struct compare_first_less_second
{
  compare_first_less_second(Compare c)
    : comp(c) {}

  template<typename T1, typename T2>
  __host__ __device__
  bool operator()(T1 lhs, T2 rhs)
  {
    return comp(thrust::get<0>(lhs), thrust::get<0>(rhs)) || (!comp(thrust::get<0>(rhs), thrust::get<0>(lhs)) && thrust::get<1>(lhs) < thrust::get<1>(rhs));
  }

  Compare comp;
}; // end compare_first_less_second


} // end namespace detail
} // end namespace thrust

