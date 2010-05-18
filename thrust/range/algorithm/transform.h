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

namespace thrust
{

namespace experimental
{

namespace range
{

namespace detail
{

template<typename SinglePassRange>
  struct transform_result
{
  typedef iterator_range<typename range_iterator<SinglePassRange>::type> type;
};


template<typename T>
  struct is_adaptable_unary_function
    : thrust::detail::is_same<
        typename T::argument_type,
        typename T::argument_type
      >
{
};


template<typename T>
  struct is_adaptable_binary_function
    : thrust::detail::is_same<
        typename T::first_argument_type,
        typename T::first_argument_type
      >
{
};


template<typename AdaptableUnaryFunction>
  struct ensure_adaptable_unary_function
    : thrust::detail::enable_if<
        is_adaptable_unary_function<
          AdaptableUnaryFunction
        >::value
      >
{
};


template<typename AdaptableBinaryFunction>
  struct ensure_adaptable_binary_function
    : thrust::detail::enable_if<
        is_adaptable_binary_function<
          AdaptableBinaryFunction
        >::value
      >
{
};


} // end detail


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


// XXX these overloads differ from Boost.Range's transform in that they take their arguments
//     as ranges and return a range of the unconsumed portion of result 
template<typename SinglePassRange1, typename SinglePassRange2, typename UnaryFunction>
  inline typename detail::transform_result<SinglePassRange2>::type
    transform(const SinglePassRange1 &rng,
              SinglePassRange2 &result,
              UnaryFunction f)
{
  typedef typename detail::transform_result<SinglePassRange2>::type Result;

  return Result(thrust::transform(begin(rng), end(rng), begin(result), f), end(result));
} // end transform()


template<typename SinglePassRange1, typename SinglePassRange2, typename SinglePassRange3, typename BinaryFunction>
  inline typename detail::transform_result<SinglePassRange3>::type
    transform(const SinglePassRange1 &rng1,
              const SinglePassRange2 &rng2,
              SinglePassRange3 &result,
              BinaryFunction f)
{
  typedef typename detail::transform_result<SinglePassRange3>::type Result;

  return Result(thrust::transform(begin(rng1), end(rng1), begin(rng2), begin(result), f), end(result));
} // end transform()


// deferred versions

namespace detail
{

template<typename SinglePassRange, typename AdaptableUnaryFunction>
  struct deferred_unary_transform_result
{
  private:
    typedef typename range_iterator<SinglePassRange>::type Iterator;
    typedef transform_iterator<AdaptableUnaryFunction,Iterator> XfrmIterator;

  public:
    typedef iterator_range<XfrmIterator> type;
}; // end deferred_unary_transform_result


template<typename AdaptableBinaryFunction>
  struct unary_function_from_binary_function
    : thrust::unary_function<
        tuple<
          typename AdaptableBinaryFunction::first_argument_type,
          typename AdaptableBinaryFunction::second_argument_type
        >,
        typename AdaptableBinaryFunction::result_type
      >
{
  typedef thrust::unary_function<
    tuple<
      typename AdaptableBinaryFunction::first_argument_type,
      typename AdaptableBinaryFunction::second_argument_type
    >,
    typename AdaptableBinaryFunction::result_type
  > super_t;

  __host__ __device__
  typename super_t::result_type
  operator()(typename super_t::argument_type x) const
  {
    return f(get<0>(x), get<1>(x));
  }

  __host__ __device__
  unary_function_from_binary_function(AdaptableBinaryFunction func)
    : f(func) {}

  AdaptableBinaryFunction f;
};


template<typename SinglePassRange1, typename SinglePassRange2, typename AdaptableBinaryFunction>
  struct deferred_binary_transform_result
{
  private:
    typedef unary_function_from_binary_function<AdaptableBinaryFunction> UnaryFunction;
    typedef typename thrust::experimental::detail::zip2_result<SinglePassRange1,SinglePassRange2>::type ZipRange;

  public:
    typedef typename deferred_unary_transform_result<ZipRange, UnaryFunction>::type type;
};


} // end detail


// XXX relax AdaptableUnaryFunction to UnaryFunction upon addition of decltype
template<typename SinglePassRange, typename AdaptableUnaryFunction>
  inline typename detail::deferred_unary_transform_result<const SinglePassRange, AdaptableUnaryFunction>::type
    transform(const SinglePassRange &rng,
              AdaptableUnaryFunction f)
{
  typedef typename detail::deferred_unary_transform_result<const SinglePassRange, AdaptableUnaryFunction>::type Result;

  return Result(make_transform_iterator(begin(rng), f), make_transform_iterator(end(rng), f));
} // end transform()


//template<typename SinglePassRange1, typename SinglePassRange2, typename AdaptableBinaryFunction>
//  inline typename detail::deferred_binary_transform_result<const SinglePassRange1, const SinglePassRange2, AdaptableBinaryFunction>::type
//    transform(const SinglePassRange1 &rng1,
//              const SinglePassRange2 &rng2,
//              AdaptableBinaryFunction f,
//              
//              typename detail::ensure_adaptable_binary_function<AdaptableBinaryFunction>::type * = 0)
//{
//  typedef detail::unary_function_from_binary_function<AdaptableBinaryFunction> Func;
//  Func unary_f(f);
//
//  typename detail::deferred_binary_transform_result<const SinglePassRange1, const SinglePassRange2, AdaptableBinaryFunction>::type Result;
//
//  return Result(transform(zip(rng1,rng2), unary_f));
//} // end transform()

} // end range

} // end experimental

} // end thrust


