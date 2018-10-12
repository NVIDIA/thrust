///////////////////////////////////////////////////////////////////////////////
//  Copyright (c)      2018 NVIDIA Corporation
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

/*! \file is_trivially_relocatable.h
 *  \brief <a href="https://wg21.link/P1144R0">P1144R0</a>'s
 *         \c is_trivially_relocatable, an extensible type trait indicating
 *         whether a type can be bitwise copied (e.g. via \c memcpy).
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/static_assert.h>
#include <thrust/detail/type_traits.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

#if THRUST_CPP_DIALECT >= 2011
  #include <type_traits>
#endif

THRUST_BEGIN_NS

namespace detail
{

template <typename T>
struct is_trivially_relocatable_impl;

} // namespace detail

/// Unary metafunction returns \c true_type if \c T is trivially relocatable, 
/// e.g. can be bitwise copied (with a facility like \c memcpy), and \c false
/// otherwise.
template <typename T>
#if THRUST_CPP_DIALECT >= 2011
using is_trivially_relocatable =
#else
struct is_trivially_relocatable :
#endif
  detail::is_trivially_relocatable_impl<T>
#if THRUST_CPP_DIALECT < 2011
{}
#endif
;

#if THRUST_CPP_DIALECT >= 2014
/// <code>constexpr bool</code> that is \c true if \c T is trivially relocatable, 
/// e.g. can be copied bitwise (with a facility like \c memcpy), and \c false
/// otherwise.
template <typename T>
constexpr bool is_trivially_relocatable_v = is_trivially_relocatable<T>::value;
#endif

/// Unary metafunction returns \c true_type if \c From is trivially relocatable
/// to \c To, e.g. can be bitwise copied (with a facility like \c memcpy), and
/// \c false otherwise.
template <typename From, typename To>
#if THRUST_CPP_DIALECT >= 2011
using is_trivially_relocatable_to =
#else
struct is_trivially_relocatable_to :
#endif
  integral_constant<
    bool
  , detail::is_same<From, To>::value && is_trivially_relocatable<To>::value
  >
#if THRUST_CPP_DIALECT < 2011
{}
#endif
;

#if THRUST_CPP_DIALECT >= 2014
/// <code>constexpr bool</code> that is \c true if \c From is trivially
/// relocatable to \c To, e.g. can be copied bitwise (with a facility like \c
/// memcpy), and \c false otherwise.
template <typename From, typename To>
constexpr bool is_trivially_relocatable_to_v
  = is_trivially_relocatable_to<From, To>::value;
#endif

/// Unary metafunction that is \c true if the element type of
/// \c FromIterator is trivially relocatable to the element type of
/// \c ToIterator.
template <typename FromIterator, typename ToIterator>
#if THRUST_CPP_DIALECT >= 2011
using is_trivially_relocatable_sequence_copy =
#else
struct is_trivially_relocatable_sequence_copy :
#endif
  integral_constant<
    bool
  ,    is_contiguous_iterator<FromIterator>::value
    && is_contiguous_iterator<ToIterator>::value
    && is_trivially_relocatable_to<
         typename thrust::iterator_traits<FromIterator>::value_type,
         typename thrust::iterator_traits<ToIterator>::value_type
       >::value
  >
#if THRUST_CPP_DIALECT < 2011
{}
#endif
;

#if THRUST_CPP_DIALECT >= 2014
/// <code>constexpr bool</code> that is \c true if the element type of
/// \c FromIterator is trivially relocatable to the element type of
/// \c ToIterator.
template <typename FromIterator, typename ToIterator>
constexpr bool is_trivial_relocatable_sequence_copy_v
  = is_trivially_relocatable_sequence_copy<FromIterator, ToIterator>::value;
#endif

/// Customization point that can be customized to indicate that a type \c T is
/// \a TriviallyRelocatable.
template <typename T>
struct proclaim_trivially_relocatable : false_type {};

///////////////////////////////////////////////////////////////////////////////

namespace detail
{

// https://wg21.link/P1144R0#wording-inheritance
template <typename T>
struct is_trivially_relocatable_impl
  : integral_constant<
      bool
      #if    THRUST_CPP_DIALECT >= 2011                                       \
          && (  (THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_GCC)            \
             || (THRUST_GCC_VERSION >= 50000))
    ,    std::is_trivially_copyable<T>::value
      #else
    ,    has_trivial_assign<T>::value
      #endif
      || proclaim_trivially_relocatable<T>::value
    >
{};

template <typename T, std::size_t N>
struct is_trivially_relocatable_impl<T[N]> : is_trivially_relocatable_impl<T> {};

} // namespace detail
 
THRUST_END_NS

