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
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>

namespace thrust
{
namespace detail
{

// the base type for all of thrust's space-annotated references.
// for reasonable reference-like semantics, derived types must reimplement the following:
// 1. constructor from pointer
// 2. copy constructor
// 3. templated copy constructor from other reference
// 4. templated assignment from other reference
// 5. assignment from value_type
template<typename Derived, typename Value, typename Pointer>
  class reference_base
{
  private:
    typedef Derived derived_type;

  public:
    typedef Pointer                                            pointer;
    typedef typename thrust::detail::remove_const<Value>::type value_type;

    __host__ __device__
    explicit reference_base(const pointer &ptr);

    template<typename OtherDerived, typename OtherValue, typename OtherPointer>
    __host__ __device__
    reference_base(const reference_base<OtherDerived,OtherValue,OtherPointer> &other,
                   typename thrust::detail::enable_if_convertible<
                     typename reference_base<OtherDerived,OtherValue,OtherPointer>::pointer,
                     pointer
                   >::type * = 0);

    derived_type &operator=(const reference_base &other);

    // XXX this may need an enable_if
    template<typename OtherDerived, typename OtherValue, typename OtherPointer>
    derived_type &operator=(const reference_base<OtherDerived,OtherValue,OtherPointer> &other);

    derived_type &operator=(const value_type &x);

    __host__ __device__
    pointer operator&() const;

    __host__ __device__
    operator value_type () const;

    __host__ __device__
    void swap(derived_type &other);

    derived_type &operator++();

    value_type operator++(int);

    // XXX parameterize the type of rhs
    derived_type &operator+=(const value_type &rhs);

    derived_type &operator--();

    value_type operator--(int);

    // XXX parameterize the type of rhs
    derived_type &operator-=(const value_type &rhs);

    // XXX parameterize the type of rhs
    derived_type &operator*=(const value_type &rhs);

    // XXX parameterize the type of rhs
    derived_type &operator/=(const value_type &rhs);

    // XXX parameterize the type of rhs
    derived_type &operator%=(const value_type &rhs);

    // XXX parameterize the type of rhs
    derived_type &operator<<=(const value_type &rhs);

    // XXX parameterize the type of rhs
    derived_type &operator>>=(const value_type &rhs);

    // XXX parameterize the type of rhs
    derived_type &operator&=(const value_type &rhs);

    // XXX parameterize the type of rhs
    derived_type &operator|=(const value_type &rhs);

    // XXX parameterize the type of rhs
    derived_type &operator^=(const value_type &rhs);

  private:
    const pointer m_ptr;

    // allow access to m_ptr for other reference_bases
    template <typename OtherDerived, typename OtherValue, typename OtherPointer> friend class reference_base;

    template<typename OtherPointer>
    inline void assign_from(OtherPointer src);

    template<typename OtherPointer>
    inline void assign_from(OtherPointer src, thrust::detail::true_type spaces_are_interoperable);

    template<typename OtherPointer>
    inline void assign_from(OtherPointer src, thrust::detail::false_type spaces_are_not_interoperable);

    inline value_type convert(thrust::detail::true_type spaces_are_interoperable) const;

    inline value_type convert(thrust::detail::false_type spaces_are_not_interoperable) const;

    inline void swap(reference_base &other, thrust::detail::true_type spaces_are_interoperable);

    inline void swap(reference_base &other, thrust::detail::false_type spaces_are_not_interoperable);
}; // end reference_base
  
} // end detail
} // end thrust

#include <thrust/detail/reference_base.inl>

