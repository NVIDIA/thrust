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

#include <thrust/detail/config.h>
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
              UnaryFunction f);


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
              UnaryFunction f);


template<typename SinglePassRange1, typename SinglePassRange2, typename SinglePassRange3, typename BinaryFunction>
  inline typename detail::binary_transform_result<SinglePassRange1, SinglePassRange2, SinglePassRange3, BinaryFunction>::type
    transform(const SinglePassRange1 &rng1,
              const SinglePassRange2 &rng2,
              SinglePassRange3 &result,
              BinaryFunction f);


template<typename SinglePassRange1, typename SinglePassRange2, typename SinglePassRange3, typename UnaryFunction, typename Predicate>
  inline typename detail::unary_transform_if_result<SinglePassRange1, SinglePassRange2, SinglePassRange3, UnaryFunction, Predicate>::type
    transform_if(const SinglePassRange1 &rng,
                 const SinglePassRange2 &stencil,
                 SinglePassRange3 &result,
                 UnaryFunction f,
                 Predicate pred);

template<typename SinglePassRange1, typename SinglePassRange2, typename SinglePassRange3, typename SinglePassRange4, typename BinaryFunction, typename Predicate>
  inline typename detail::binary_transform_if_result<SinglePassRange1, SinglePassRange2, SinglePassRange3, SinglePassRange4, BinaryFunction, Predicate>::type
    transform_if(const SinglePassRange1 &rng1,
                 const SinglePassRange2 &rng2,
                 const SinglePassRange3 &stencil,
                 SinglePassRange4 &result,
                 BinaryFunction f,
                 Predicate pred);


// lazy versions

// XXX relax AdaptableUnaryFunction to UnaryFunction upon addition of decltype
template<typename SinglePassRange, typename AdaptableUnaryFunction>
  inline typename detail::lazy_unary_transform_result<const SinglePassRange, AdaptableUnaryFunction>::type
    transform(const SinglePassRange &rng,
              AdaptableUnaryFunction f);


// XXX relax AdaptableBinaryFunction to BinaryFunction upon addition of decltype
template<typename SinglePassRange1, typename SinglePassRange2, typename AdaptableBinaryFunction>
  inline typename detail::lazy_binary_transform_result<const SinglePassRange1, const SinglePassRange2, AdaptableBinaryFunction>::type
    transform(const SinglePassRange1 &rng1,
              const SinglePassRange2 &rng2,
              AdaptableBinaryFunction f);


} // end range

} // end experimental

} // end thrust

#include <thrust/range/algorithm/detail/transform.inl>


