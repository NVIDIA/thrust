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

//  Copyright Neil Groves 2009. Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//
// For more information, see http://www.boost.org/libs/range/

#pragma once

#include <thrust/range/begin.h>
#include <thrust/range/end.h>
#include <thrust/range/iterator_range.h>
#include <thrust/range/detail/iterator.h>
#include <thrust/range/zip.h>

#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>

#include <thrust/range/algorithm/detail/transform_result.h>

namespace thrust
{

namespace experimental
{

namespace range
{


// XXX should we implement transform() with transform(begin,end) or for_each(rng) ?


// XXX Boost's versions which take iterator arguments
//template<typename SinglePassRange, typename OutputIterator, typename UnaryFunction>
//  inline OutputIterator
//    transform(const SinglePassRange &rng,
//              OutputIterator result,
//              UnaryFunction f)
//{
//  return thrust::transform(begin(rng), end(rng), result, f);
//} // end transform()
//
//
//template<typename SinglePassRange1, typename SinglePassRange2, typename OutputIterator, typename UnaryFunction>
//  inline OutputIterator
//    transform(const SinglePassRange1 &rng,
//              const SinglePassRange2 &rng,
//              OutputIterator result,
//              UnaryFunction f)
//{
//  return thrust::transform(begin(rng1), end(rng1), begin(rng2), result, f);
//} // end transform()


// these overloads differ from Boost.Range's transform in that they take their arguments
// as ranges and return a range of the unconsumed portion of result
template<typename SinglePassRange1, typename SinglePassRange2, typename UnaryFunction>
  inline typename detail::unary_transform_result<SinglePassRange1, SinglePassRange2, UnaryFunction>::type
    transform(const SinglePassRange1 &rng,
              SinglePassRange2 &result,
              UnaryFunction f)
{
  typedef typename detail::unary_transform_result<SinglePassRange1, SinglePassRange2, UnaryFunction>::type Result;

  return Result(thrust::transform(begin(rng), end(rng), begin(result), f), end(result));
} // end transform()


// add a second overload to accept temporary ranges for the second parameter from things like zip()
//
// XXX change
//
//     const SinglePassRange2 &result
//
//     to
//
//     SinglePassRange2 &&result
//
//     upon addition of rvalue references
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


