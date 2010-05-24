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

namespace thrust
{

namespace experimental
{

namespace range
{


// XXX should we implement transform() with transform(begin,end) or for_each(rng) ?

template<typename SinglePassRange1, typename SinglePassRange2, typename UnaryFunction>
  inline typename detail::unary_transform_result<SinglePassRange1, SinglePassRange2, UnaryFunction>::type
    transform(const SinglePassRange1 &rng,
              SinglePassRange2 &result,
              UnaryFunction f)
{
  typedef typename detail::unary_transform_result<SinglePassRange1, SinglePassRange2, UnaryFunction>::type Result;

  return Result(thrust::transform(begin(rng), end(rng), begin(result), f), end(result));
} // end transform()


template<typename SinglePassRange1, typename SinglePassRange2, typename UnaryFunction>
  inline typename detail::unary_transform_result<SinglePassRange1, SinglePassRange2, UnaryFunction>::type
    transform(const SinglePassRange1 &rng,
              const SinglePassRange2 &result,
              UnaryFunction f)
{
  typedef typename detail::unary_transform_result<SinglePassRange1, SinglePassRange2, UnaryFunction>::type Result;

  return Result(thrust::transform(begin(rng), end(rng), begin(result), f), end(result));
} // end transform()


template<typename SinglePassRange1, typename SinglePassRange2, typename SinglePassRange3, typename BinaryFunction>
  inline typename detail::binary_transform_result<SinglePassRange1, SinglePassRange2, SinglePassRange3, BinaryFunction>::type
    transform(const SinglePassRange1 &rng1,
              const SinglePassRange2 &rng2,
              SinglePassRange3 &result,
              BinaryFunction f)
{
  typedef typename detail::binary_transform_result<SinglePassRange1, SinglePassRange2, SinglePassRange3, BinaryFunction>::type Result;

  return Result(thrust::transform(begin(rng1), end(rng1), begin(rng2), begin(result), f), end(result));
} // end transform()


template<typename SinglePassRange1, typename SinglePassRange2, typename SinglePassRange3, typename UnaryFunction, typename Predicate>
  inline typename detail::unary_transform_if_result<SinglePassRange1, SinglePassRange2, SinglePassRange3, UnaryFunction, Predicate>::type
    transform_if(const SinglePassRange1 &rng,
                 const SinglePassRange2 &stencil,
                 SinglePassRange3 &result,
                 UnaryFunction f,
                 Predicate pred)
{
  typedef typename detail::unary_transform_if_result<SinglePassRange1, SinglePassRange2, SinglePassRange3, UnaryFunction, Predicate>::type Result;
  return Result(thrust::transform_if(begin(rng), end(rng), begin(stencil), begin(result), f, pred), end(result));
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

