/*
 *  Copyright 2008-2009 NVIDIA Corporation
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
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_facade.h>

namespace thrust
{

namespace experimental
{

// forward declaration of use_default
struct use_default;

namespace detail
{

template <bool, typename Then, typename Else>
  struct eval_if
{
}; // end eval_if

template<typename Then, typename Else>
  struct eval_if<true, Then, Else>
{
  typedef typename Then::type type;
}; // end eval_if

template<typename Then, typename Else>
  struct eval_if<false, Then, Else>
{
  typedef typename Else::type type;
}; // end eval_if

template<typename T>
  struct identity
{
  typedef T type;
}; // end identity

// If T is use_default, return the result of invoking
// DefaultNullaryFn, otherwise return T.
template <class T, class DefaultNullaryFn>
struct ia_dflt_help
  : eval_if<
        thrust::detail::is_same<T, use_default>::value
      , DefaultNullaryFn
      , identity<T>
    >
{
};

template<typename Iterator>
  struct iterator_value
{
  typedef typename thrust::iterator_traits<Iterator>::value_type type;
}; // end iterator_value


template<typename Iterator>
  struct iterator_traversal
{
  typedef typename thrust::iterator_traits<Iterator>::iterator_category type;
}; // end iterator_traversal


template<typename Iterator>
  struct iterator_reference
{
  typedef typename thrust::iterator_traits<Iterator>::reference type;
}; // end iterator_reference


template<typename Iterator>
  struct iterator_difference
{
  typedef typename thrust::iterator_traits<Iterator>::difference_type type;
}; // end iterator_difference


// A metafunction which computes an iterator_adaptor's base class,
// a specialization of iterator_facade.
template <
    typename Derived
  , typename Base
  , typename Pointer
  , typename Value
  , typename Traversal
  , typename Reference
  , typename Difference
>
  struct iterator_adaptor_base
{
  typedef iterator_facade<
      Derived

    , Pointer

    , typename ia_dflt_help<
          Value
        , iterator_value<Base>
      >::type

    , typename ia_dflt_help<
          Traversal
        , iterator_traversal<Base>
      >::type

    , typename ia_dflt_help<
          Reference
        , eval_if<
            thrust::detail::is_same<Value,use_default>::value
          , iterator_reference<Base>
          , thrust::detail::add_reference<Value>
        >
      >::type

    , typename ia_dflt_help<
          Difference
        , iterator_difference<Base>
      >::type
  >
  type;
}; // end iterator_adaptor_base

} // end detail

} // end experimental

} // end thrust

