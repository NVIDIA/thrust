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

#include <thrust/detail/type_traits.h>
#include <thrust/detail/function_traits.h>
#include <thrust/range/detail/zip_result.h>

namespace thrust
{

namespace experimental
{

namespace range
{

namespace detail
{

template<typename SinglePassRange1, typename SinglePassRange2, typename UnaryFunction>
  struct unary_transform_result
    : thrust::detail::enable_if_c<
        !thrust::detail::is_adaptable_binary_function<UnaryFunction>::value,
        iterator_range<SinglePassRange2>
      >
{};


template<typename SinglePassRange1, typename SinglePassRange2, typename SinglePassRange3, typename BinaryFunction>
  struct binary_transform_result
{
  typedef iterator_range<SinglePassRange3> type;
};


template<typename SinglePassRange1, typename SinglePassRange2, typename SinglePassRange3, typename UnaryFunction, typename Predicate>
  struct unary_transform_if_result
{
  typedef iterator_range<SinglePassRange3> type;
};


template<typename SinglePassRange1, typename SinglePassRange2, typename SinglePassRange3, typename SinglePassRange4, typename BinaryFunction, typename Predicate>
  struct binary_transform_if_result
{
  typedef iterator_range<SinglePassRange4> type;
};


template<typename SinglePassRange, typename AdaptableUnaryFunction>
  struct lazy_unary_transform_result
    : thrust::detail::lazy_enable_if<
        thrust::detail::is_adaptable_unary_function<AdaptableUnaryFunction>,
        thrust::detail::identity_<
          iterator_range<
            transform_iterator<
              AdaptableUnaryFunction,
              typename range_iterator<SinglePassRange>::type
            >
          >
        >
      >
{
}; // end lazy_unary_transform_result


template<typename AdaptableBinaryFunction>
  struct unary_function_of_tuple_from_binary_function
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
  unary_function_of_tuple_from_binary_function(void) {}

  __host__ __device__
  unary_function_of_tuple_from_binary_function(AdaptableBinaryFunction func)
    : f(func) {}

  AdaptableBinaryFunction f;
}; // end unary_function_of_tuple_from_binary_function


template<typename SinglePassRange1, typename SinglePassRange2, typename AdaptableBinaryFunction>
  struct lazy_binary_transform_result
    : thrust::detail::lazy_enable_if<
        thrust::detail::is_adaptable_binary_function<AdaptableBinaryFunction>,
        lazy_unary_transform_result<
          typename thrust::experimental::detail::zip2_result<
            SinglePassRange1,
            SinglePassRange2
          >::type,
          unary_function_of_tuple_from_binary_function<AdaptableBinaryFunction>
        >
      >
{}; // end lazy_binary_transform_result


} // end detail

} // end range

} // end experimental

} // end thrust

