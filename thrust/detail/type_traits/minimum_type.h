/*
 *  Copyright 2008-2011 NVIDIA Corporation
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

namespace minimum_type_detail
{

//
// Returns the minimum type or causes an error
// if T1 and T2 are unrelated.
//
template <typename T1, typename T2, bool GreaterEqual, bool LessEqual> struct minimum_type_impl;
  
template <typename T1, typename T2>
struct minimum_type_impl<T1,T2,true,false>
{
  typedef T2 type;
}; // end minimum_type_impl

template <typename T1, typename T2>
struct minimum_type_impl<T1,T2,false,true>
{
  typedef T1 type;
}; // end minimum_type_impl

template <typename T1, typename T2>
struct minimum_type_impl<T1,T2,true,true>
{
  typedef T1 type;
}; // end minimum_type_impl

template <class T1, class T2>
struct primitive_minimum_type
  : minimum_type_detail::minimum_type_impl<
      T1,
      T2,
      ::thrust::detail::is_convertible<T1,T2>::value,
      ::thrust::detail::is_convertible<T2,T1>::value
    >
{
}; // end primitive_minimum_type

// XXX this belongs somewhere more general
struct any_conversion
{
  template<typename T> operator T (void);
};

} // end minimum_type_detail

template<typename T1,
         typename T2  = minimum_type_detail::any_conversion,
         typename T3  = minimum_type_detail::any_conversion,
         typename T4  = minimum_type_detail::any_conversion,
         typename T5  = minimum_type_detail::any_conversion,
         typename T6  = minimum_type_detail::any_conversion,
         typename T7  = minimum_type_detail::any_conversion,
         typename T8  = minimum_type_detail::any_conversion,
         typename T9  = minimum_type_detail::any_conversion,
         typename T10 = minimum_type_detail::any_conversion,
         typename T11 = minimum_type_detail::any_conversion,
         typename T12 = minimum_type_detail::any_conversion,
         typename T13 = minimum_type_detail::any_conversion,
         typename T14 = minimum_type_detail::any_conversion,
         typename T15 = minimum_type_detail::any_conversion,
         typename T16 = minimum_type_detail::any_conversion>
  class minimum_type
{
  typedef typename minimum_type_detail::primitive_minimum_type<T1,T2>::type    type1;
  typedef typename minimum_type_detail::primitive_minimum_type<T3,T4>::type    type2;
  typedef typename minimum_type_detail::primitive_minimum_type<T5,T6>::type    type3;
  typedef typename minimum_type_detail::primitive_minimum_type<T7,T8>::type    type4;
  typedef typename minimum_type_detail::primitive_minimum_type<T9,T10>::type   type5;
  typedef typename minimum_type_detail::primitive_minimum_type<T11,T12>::type  type6;
  typedef typename minimum_type_detail::primitive_minimum_type<T13,T14>::type  type7;
  typedef typename minimum_type_detail::primitive_minimum_type<T15,T16>::type  type8;

  typedef typename minimum_type_detail::primitive_minimum_type<type1,type2>::type    type9;
  typedef typename minimum_type_detail::primitive_minimum_type<type3,type4>::type    type10;
  typedef typename minimum_type_detail::primitive_minimum_type<type5,type6>::type    type11;
  typedef typename minimum_type_detail::primitive_minimum_type<type7,type8>::type    type12;

  typedef typename minimum_type_detail::primitive_minimum_type<type9,type10>::type   type13;
  typedef typename minimum_type_detail::primitive_minimum_type<type11,type12>::type  type14;

  public:
    typedef typename minimum_type_detail::primitive_minimum_type<type13, type14>::type type;
}; // end minimum_type

} // end detail

} // end thrust

