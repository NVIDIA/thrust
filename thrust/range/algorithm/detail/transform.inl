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

#pragma once

#include <thrust/range/algorithm/transform.h>
#include <thrust/range/algorithm/detail/transform_result.h>
#include <thrust/range/algorithm/for_each.h>
#include <thrust/range/zip.h>
#include <thrust/range/iterator_range.h>
#include <thrust/tuple.h>


namespace thrust
{

namespace experimental
{

namespace range
{

namespace detail
{

// XXX these functors are duplicated from the ones in thrust/detail/device/generic/transform.inl
//     put them some place common, maybe detail/functional.h
template <typename UnaryFunction>
struct unary_transform_functor
{
  UnaryFunction f;

  unary_transform_functor(UnaryFunction f_)
    :f(f_)
  {}

  template<typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<1>(t) = f(thrust::get<0>(t));
  }
}; // end unary_transform_functor


template <typename BinaryFunction>
struct binary_transform_functor
{
  BinaryFunction f;

  binary_transform_functor(BinaryFunction f_)
    :f(f_)
  {}

  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  { 
    thrust::get<2>(t) = f(thrust::get<0>(t), thrust::get<1>(t));
  }
}; // end binary_transform_functor


template <typename UnaryFunction, typename Predicate>
struct unary_transform_if_functor
{
  UnaryFunction unary_op;
  Predicate pred;
  
  unary_transform_if_functor(UnaryFunction _unary_op, Predicate _pred)
    : unary_op(_unary_op), pred(_pred) {} 
  
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    if(pred(thrust::get<1>(t)))
      thrust::get<2>(t) = unary_op(thrust::get<0>(t));
  }
}; // end unary_transform_if_functor


template <typename BinaryFunction, typename Predicate>
struct binary_transform_if_functor
{
  BinaryFunction binary_op;
  Predicate pred;

  binary_transform_if_functor(BinaryFunction _binary_op, Predicate _pred)
    : binary_op(_binary_op), pred(_pred) {} 

  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    if(pred(thrust::get<2>(t)))
      thrust::get<3>(t) = binary_op(thrust::get<0>(t), thrust::get<1>(t));
  }
}; // end binary_transform_if_functor
  

} // end detail


template<typename SinglePassRange1, typename SinglePassRange2, typename UnaryFunction>
  inline typename detail::unary_transform_result<SinglePassRange1, SinglePassRange2, UnaryFunction>::type
    transform(const SinglePassRange1 &rng,
              SinglePassRange2 &result,
              UnaryFunction f)
{
  for_each(zip(rng,result), detail::unary_transform_functor<UnaryFunction>(f));

  // XXX begin() + size() isn't generic
  return make_iterator_range(begin(result) + rng.size(), end(result));
} // end transform()


template<typename SinglePassRange1, typename SinglePassRange2, typename UnaryFunction>
  inline typename detail::unary_transform_result<SinglePassRange1, SinglePassRange2, UnaryFunction>::type
    transform(const SinglePassRange1 &rng,
              const SinglePassRange2 &result,
              UnaryFunction f)
{
  for_each(zip(rng,result), detail::unary_transform_functor<UnaryFunction>(f));

  // XXX begin() + size() isn't generic
  return make_iterator_range(begin(result) + rng.size(), end(result));
} // end transform()


template<typename SinglePassRange1, typename SinglePassRange2, typename SinglePassRange3, typename BinaryFunction>
  inline typename detail::binary_transform_result<SinglePassRange1, SinglePassRange2, SinglePassRange3, BinaryFunction>::type
    transform(const SinglePassRange1 &rng1,
              const SinglePassRange2 &rng2,
              SinglePassRange3 &result,
              BinaryFunction f)
{
  for_each(zip(rng1,rng2,result), detail::binary_transform_functor<BinaryFunction>(f));

  // XXX begin() + size() isn't generic
  return make_iterator_range(begin(result) + rng1.size(), end(result));
} // end transform()


template<typename SinglePassRange1, typename SinglePassRange2, typename SinglePassRange3, typename UnaryFunction, typename Predicate>
  inline typename detail::unary_transform_if_result<SinglePassRange1, SinglePassRange2, SinglePassRange3, UnaryFunction, Predicate>::type
    transform_if(const SinglePassRange1 &rng,
                 const SinglePassRange2 &stencil,
                 SinglePassRange3 &result,
                 UnaryFunction f,
                 Predicate pred)
{
  for_each(zip(rng,stencil,result), detail::unary_transform_if_functor<UnaryFunction,Predicate>(f, pred));

  // XXX begin() + size() isn't generic
  return make_iterator_range(begin(result) + rng.size(), end(result));
} // end transform_if()


template<typename SinglePassRange1, typename SinglePassRange2, typename SinglePassRange3, typename SinglePassRange4, typename BinaryFunction, typename Predicate>
  inline typename detail::binary_transform_if_result<SinglePassRange1, SinglePassRange2, SinglePassRange3, SinglePassRange4, BinaryFunction, Predicate>::type
    transform_if(const SinglePassRange1 &rng1,
                 const SinglePassRange2 &rng2,
                 const SinglePassRange3 &stencil,
                 SinglePassRange4 &result,
                 BinaryFunction f,
                 Predicate pred)
{
  for_each(zip(rng1,rng2,stencil,result), detail::binary_transform_if_functor<BinaryFunction,Predicate>(f, pred));

  // XXX begin() + size() isn't generic
  return make_iterator_range(begin(result) + rng1.size(), end(result));
} // end transform_if()


// lazy versions

// XXX relax AdaptableUnaryFunction to UnaryFunction upon addition of decltype
template<typename SinglePassRange, typename AdaptableUnaryFunction>
  inline typename detail::lazy_unary_transform_result<const SinglePassRange, AdaptableUnaryFunction>::type
    transform(const SinglePassRange &rng,
              AdaptableUnaryFunction f)
{
  typedef typename detail::lazy_unary_transform_result<const SinglePassRange, AdaptableUnaryFunction>::type Result;

  return Result(make_transform_iterator(begin(rng), f), make_transform_iterator(end(rng), f));
} // end transform()


// XXX relax AdaptableBinaryFunction to BinaryFunction upon addition of decltype
template<typename SinglePassRange1, typename SinglePassRange2, typename AdaptableBinaryFunction>
  inline typename detail::lazy_binary_transform_result<const SinglePassRange1, const SinglePassRange2, AdaptableBinaryFunction>::type
    transform(const SinglePassRange1 &rng1,
              const SinglePassRange2 &rng2,
              AdaptableBinaryFunction f)
{
  return transform(zip(rng1,rng2), detail::unary_function_of_tuple_from_binary_function<AdaptableBinaryFunction>(f));
} // end transform()


} // end range

} // end experimental

} // end thrust

