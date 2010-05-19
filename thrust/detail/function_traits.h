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

namespace thrust
{

namespace detail
{


// XXX replicating this idiom will get old quick
//     we need a __THRUST_DECLARE_HAS_NESTED_TYPE or something

template<typename T>
  struct has_result_type
{
  typedef char yes_type;
  typedef int  no_type;

  template<typename S> static yes_type test(typename S::result_type *);

  template<typename S> static no_type test(...);

  static bool const value = sizeof(test<T>(0)) == sizeof(yes_type);
  typedef thrust::detail::integral_constant<bool, value> type;
}; // end has_result_type


template<typename T>
  struct has_argument_type
{
  typedef char yes_type;
  typedef int  no_type;

  template<typename S> static yes_type test(typename S::argument_type *);

  template<typename S> static no_type test(...);

  static bool const value = sizeof(test<T>(0)) == sizeof(yes_type);
  typedef thrust::detail::integral_constant<bool, value> type;
}; // end has_argument_type


template<typename T>
  struct has_first_argument_type
{
  typedef char yes_type;
  typedef int  no_type;

  template<typename S> static yes_type test(typename S::first_argument_type *);

  template<typename S> static no_type test(...);

  static bool const value = sizeof(test<T>(0)) == sizeof(yes_type);
  typedef thrust::detail::integral_constant<bool, value> type;
}; // end has_first_argument_type


template<typename T>
  struct has_second_argument_type
{
  typedef char yes_type;
  typedef int  no_type;

  template<typename S> static yes_type test(typename S::second_argument_type *);

  template<typename S> static no_type test(...);

  static bool const value = sizeof(test<T>(0)) == sizeof(yes_type);
  typedef thrust::detail::integral_constant<bool, value> type;
}; // end has_second_argument_type


template<typename T>
  struct is_adaptable_unary_function
    : thrust::detail::and_<
        has_result_type<T>,
        has_argument_type<T>
      >
{};


template<typename T>
  struct is_adaptable_binary_function
    : thrust::detail::and_<
        has_result_type<T>,
        thrust::detail::and_<
          has_first_argument_type<T>,
          has_second_argument_type<T>
        >
      >
{};


} // end detail

} // end thrust

