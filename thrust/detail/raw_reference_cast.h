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

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits/has_nested_type.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/tuple_of_iterator_references.h>

namespace thrust
{
namespace detail
{


__THRUST_DEFINE_HAS_NESTED_TYPE(is_wrapped_reference, wrapped_reference_hint)

namespace raw_reference_detail
{

template<typename T, typename Enable = void>
  struct raw_reference
    : add_reference<T>
{};


// XXX consider making raw_reference<T&> an error


template<typename T>
  struct raw_reference<
    T,
    typename thrust::detail::enable_if<
      is_wrapped_reference<
        typename remove_cv<T>::type
      >::value
    >::type
  >
{
  typedef typename add_reference<
    typename pointer_element<typename T::pointer>::type
  >::type type;
};

} // end raw_reference_ns

template<typename T>
  struct raw_reference : 
    raw_reference_detail::raw_reference<T>
{};


// wrapped reference-like things which aren't strictly wrapped references
// (e.g. tuples of wrapped references) are considered unwrappable
template<typename T>
  struct is_unwrappable
    : is_wrapped_reference<T>
{};


template<typename T, typename Result = void>
  struct enable_if_unwrappable
    : enable_if<
        is_unwrappable<T>::value,
        Result
      >
{};


} // end detail


template<typename T>
  inline __host__ __device__ typename detail::raw_reference<T>::type raw_reference_cast(T &ref)
{
  return *thrust::raw_pointer_cast(&ref);
} // end raw_reference_cast


template<typename T>
  inline __host__ __device__ typename detail::raw_reference<const T>::type raw_reference_cast(const T &ref)
{
  return *thrust::raw_pointer_cast(&ref);
} // end raw_reference_cast


template<
  typename T0, typename T1, typename T2,
  typename T3, typename T4, typename T5,
  typename T6, typename T7, typename T8,
  typename T9
>
inline __host__ __device__
typename detail::enable_if_unwrappable<
  thrust::detail::tuple_of_iterator_references<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>,
  typename detail::raw_reference<
    thrust::detail::tuple_of_iterator_references<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
  >::type
>::type
raw_reference_cast(detail::tuple_of_iterator_references<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9> t);


} // end thrust

#include <thrust/detail/raw_reference_cast.inl>

