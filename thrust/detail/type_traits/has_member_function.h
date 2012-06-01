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

#pragma once

#include <thrust/detail/type_traits.h>

#define __THRUST_DEFINE_HAS_MONOMORPHIC_MEMBER_FUNCTION(trait_name, member_function_name) \
template<typename T> \
  struct trait_name  \
{                    \
  typedef char yes_type;                          \
  typedef struct {yes_type array[2];} no_type;    \
  template<typename S> static yes_type test(char (*)[sizeof(&S::member_function_name)]); \
  template<typename S> static no_type  test(...); \
  static bool const value = sizeof(test<T>(0)) == sizeof(yes_type); \
  typedef thrust::detail::integral_constant<bool, value> type; \
};

#define __THRUST_DEFINE_HAS_MEMBER_FUNCTION0(trait_name, member_function_name) \
template<typename T, typename Result> \
  struct trait_name                   \
{                                     \
  typedef char yes_type;                       \
  typedef struct {yes_type array[2];} no_type; \
  template<typename S, Result(S::*)()> struct check; \
  template<typename S> static yes_type test(check<S,&S::member_function_name> *); \
  template<typename S> static no_type  test(...); \
  static bool const value = sizeof(test<T>(0)) == sizeof(yes_type); \
  typedef thrust::detail::integral_constant<bool,value> type; \
};

#define __THRUST_DEFINE_HAS_MEMBER_FUNCTION1(trait_name, member_function_name) \
template<typename T, typename Result, typename Arg>     \
  struct trait_name                                     \
{                                                       \
  typedef char yes_type;                                \
  typedef struct {yes_type array[2];} no_type;          \
  template<typename S, Result(S::*)(Arg)> struct check; \
  template<typename S> static yes_type test(check<S,&S::member_function_name> *); \
  template<typename S> static no_type  test(...); \
  static bool const value = sizeof(test<T>(0)) == sizeof(yes_type); \
  typedef thrust::detail::integral_constant<bool,value> type; \
};

#define __THRUST_DEFINE_HAS_MEMBER_FUNCTION2(trait_name, member_function_name) \
template<typename T, typename Result, typename Arg1, typename Arg2>   \
  struct trait_name                                                   \
{                                                                     \
  typedef char yes_type;                                              \
  typedef struct {yes_type array[2];} no_type;                        \
  template<typename S, Result(S::*)(Arg1,Arg2)> struct check;         \
  template<typename S> static yes_type test(check<S,&S::member_function_name> *); \
  template<typename S> static no_type  test(...); \
  static bool const value = sizeof(test<T>(0)) == sizeof(yes_type); \
  typedef thrust::detail::integral_constant<bool,value> type; \
};

#define __THRUST_DEFINE_HAS_MEMBER_FUNCTION3(trait_name, member_function_name)     \
template<typename T, typename Result, typename Arg1, typename Arg2, typename Arg3> \
  struct trait_name                                                                \
{                                                                                  \
  typedef char yes_type;                                                           \
  typedef struct {yes_type array[2];} no_type;                                     \
  template<typename S, Result(S::*)(Arg1,Arg2,Arg3)> struct check;                 \
  template<typename S> static yes_type test(check<S,&S::member_function_name> *);  \
  template<typename S> static no_type  test(...); \
  static bool const value = sizeof(test<T>(0)) == sizeof(yes_type); \
  typedef thrust::detail::integral_constant<bool,value> type; \
};

