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

// forward references to lambda placeholders defined in zip_iterator.inl
struct _1;
struct _2;

//
// Returns the minimum category type or error_type
// if T1 and T2 are unrelated.
//
template <bool GreaterEqual, bool LessEqual>
struct minimum_category_impl
//# if BOOST_WORKAROUND(BOOST_MSVC, < 1300)
//{
//    template <class T1, class T2> struct apply
//    {
//        typedef T2 type;
//    };
//    typedef void type;
//}
//# endif 
;

template <class T1, class T2>
struct error_not_related_by_convertibility;
  
template <>
struct minimum_category_impl<true,false>
{
  template <class T1, class T2> struct apply
  {
    typedef T2 type;
  }; // end apply
}; // end minimum_category_impl

template <>
struct minimum_category_impl<false,true>
{
  template <class T1, class T2> struct apply
  {
    typedef T1 type;
  }; // end apply
}; // end minimum_category_impl

template <>
struct minimum_category_impl<true,true>
{
  template <class T1, class T2> struct apply
  {
    //BOOST_STATIC_ASSERT((is_same<T1,T2>::value));
    typedef T1 type;
  }; // end apply
}; // end minimum_category_impl

template <>
struct minimum_category_impl<false,false>
{
  template <class T1, class T2> struct apply
    : error_not_related_by_convertibility<T1,T2>
  {
  }; // end apply
}; // end minimum_category_impl

template <class T1 = _1, class T2 = _2>
struct minimum_category
{
  typedef minimum_category_impl< 
      ::thrust::detail::is_convertible<T1,T2>::value
    , ::thrust::detail::is_convertible<T2,T1>::value
  > outer;
  
  typedef typename outer::template apply<T1,T2> inner;
  typedef typename inner::type type;
    
  //BOOST_MPL_AUX_LAMBDA_SUPPORT(2,minimum_category,(T1,T2))
}; // end minimum_category
    
template <>
struct minimum_category<_1,_2>
{
  template <class T1, class T2>
  struct apply : minimum_category<T1,T2>
  {};
  
  //BOOST_MPL_AUX_LAMBDA_SUPPORT_SPEC(2,minimum_category,(_1,_2))
}; // end minimum_category

} // end detail

} // end thrust

