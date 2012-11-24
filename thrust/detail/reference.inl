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

#include <thrust/detail/config.h>
#include <thrust/detail/reference.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/memory.h>
#include <thrust/system/detail/adl/get_value.h>
#include <thrust/system/detail/adl/assign_value.h>
#include <thrust/system/detail/adl/iter_swap.h>


namespace thrust
{


template<typename Element, typename Pointer, typename Derived>
  template<typename OtherElement, typename OtherPointer, typename OtherDerived>
    reference<Element,Pointer,Derived>
      ::reference(const reference<OtherElement,OtherPointer,OtherDerived> &other,
                  typename thrust::detail::enable_if_convertible<
                    typename reference<OtherElement,OtherPointer,OtherDerived>::pointer,
                    pointer
                  >::type *)
        : m_ptr(other.m_ptr)
{}


template<typename Element, typename Pointer, typename Derived>
  reference<Element,Pointer,Derived>
    ::reference(const pointer &ptr)
      : m_ptr(ptr)
{}


template<typename Element, typename Pointer, typename Derived>
  typename reference<Element,Pointer,Derived>::pointer
    reference<Element,Pointer,Derived>
      ::operator&() const
{
  return m_ptr;
} // end reference::operator&()


template<typename Element, typename Pointer, typename Derived>
  typename reference<Element,Pointer,Derived>::derived_type &
    reference<Element,Pointer,Derived>
      ::operator=(const value_type &v)
{
  assign_from(&v);
  return static_cast<derived_type&>(*this);
} // end reference::operator=()


template<typename Element, typename Pointer, typename Derived>
  typename reference<Element,Pointer,Derived>::derived_type &
    reference<Element,Pointer,Derived>
      ::operator=(const reference &other)
{
  assign_from(&other); 
  return static_cast<derived_type&>(*this);
} // end reference::operator=()


template<typename Element, typename Pointer, typename Derived>
  template<typename OtherElement, typename OtherPointer, typename OtherDerived>
    typename reference<Element,Pointer,Derived>::derived_type &
      reference<Element,Pointer,Derived>
        ::operator=(const reference<OtherElement,OtherPointer,OtherDerived> &other)
{
  assign_from(&other);
  return static_cast<derived_type&>(*this);
} // end reference::operator=()


template<typename Element, typename Pointer, typename Derived>
  template<typename System>
    typename reference<Element,Pointer,Derived>::value_type
      reference<Element,Pointer,Derived>
        ::convert_to_value_type(System *system) const
{
  using thrust::system::detail::generic::select_system;
  return strip_const_get_value(select_system(*system));
} // end convert_to_value_type()


template<typename Element, typename Pointer, typename Derived>
  reference<Element,Pointer,Derived>
    ::operator typename reference<Element,Pointer,Derived>::value_type () const
{
  typedef typename thrust::iterator_system<pointer>::type System;

  // XXX avoid default-constructing a system
  // XXX use null a reference for dispatching
  // XXX this assumes that the eventual invocation of
  // XXX get_value will not access system state
  System *system = 0;

  return convert_to_value_type(system);
} // end reference::operator value_type ()


template<typename Element, typename Pointer, typename Derived>
  template<typename System>
    typename reference<Element,Pointer,Derived>::value_type
      reference<Element,Pointer,Derived>
        ::strip_const_get_value(const System &system) const
{
  System &non_const_system = const_cast<System&>(system);

  using thrust::system::detail::generic::get_value;

  return get_value(thrust::detail::derived_cast(non_const_system), m_ptr);
} // end reference::strip_const_get_value()


template<typename Element, typename Pointer, typename Derived>
  template<typename System1, typename System2, typename OtherPointer>
    void reference<Element,Pointer,Derived>
      ::assign_from(System1 *system1, System2 *system2, OtherPointer src)
{
  using thrust::system::detail::generic::select_system;

  strip_const_assign_value(select_system(*system1, *system2), src);
} // end assign_from()


template<typename Element, typename Pointer, typename Derived>
  template<typename OtherPointer>
    void reference<Element,Pointer,Derived>
      ::assign_from(OtherPointer src)
{
  typedef typename thrust::iterator_system<pointer>::type      System1;
  typedef typename thrust::iterator_system<OtherPointer>::type System2;

  // XXX avoid default-constructing a system
  // XXX use null references for dispatching
  // XXX this assumes that the eventual invocation of
  // XXX assign_value will not access system state
  System1 *system1 = 0;
  System2 *system2 = 0;

  assign_from(system1, system2, src);
} // end assign_from()


template<typename Element, typename Pointer, typename Derived>
  template<typename System, typename OtherPointer>
    void reference<Element,Pointer,Derived>
      ::strip_const_assign_value(const System &system, OtherPointer src)
{
  System &non_const_system = const_cast<System&>(system);

  using thrust::system::detail::generic::assign_value;

  assign_value(thrust::detail::derived_cast(non_const_system), m_ptr, src);
} // end strip_const_assign_value()


template<typename Element, typename Pointer, typename Derived>
  template<typename System>
    void reference<Element,Pointer,Derived>
      ::swap(System *system, derived_type &other)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::iter_swap;

  iter_swap(select_system(*system, *system), m_ptr, other.m_ptr);
} // end reference::swap()


template<typename Element, typename Pointer, typename Derived>
  void reference<Element,Pointer,Derived>
    ::swap(derived_type &other)
{
  typedef typename thrust::iterator_system<pointer>::type System;

  // XXX avoid default-constructing a system
  // XXX use null references for dispatching
  // XXX this assumes that the eventual invocation
  // XXX of iter_swap will not access system state
  System *system = 0;

  swap(system, other);
} // end reference::swap()


template<typename Element, typename Pointer, typename Derived>
  typename reference<Element,Pointer,Derived>::derived_type &
    reference<Element,Pointer,Derived>
      ::operator++(void)
{
  value_type temp = *this;
  ++temp;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference::operator++()


template<typename Element, typename Pointer, typename Derived>
  typename reference<Element,Pointer,Derived>::value_type
    reference<Element,Pointer,Derived>
      ::operator++(int)
{
  value_type temp = *this;
  value_type result = temp++;
  *this = temp;
  return result;
} // end reference::operator++()


template<typename Element, typename Pointer, typename Derived>
  typename reference<Element,Pointer,Derived>::derived_type &
    reference<Element,Pointer,Derived>
      ::operator+=(const value_type &rhs)
{
  value_type temp = *this;
  temp += rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference::operator+=()

template<typename Element, typename Pointer, typename Derived>
  typename reference<Element,Pointer,Derived>::derived_type &
    reference<Element,Pointer,Derived>
      ::operator--(void)
{
  value_type temp = *this;
  --temp;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference::operator--()

template<typename Element, typename Pointer, typename Derived>
  typename reference<Element,Pointer,Derived>::value_type
    reference<Element,Pointer,Derived>
      ::operator--(int)
{
  value_type temp = *this;
  value_type result = temp--;
  *this = temp;
  return result;
} // end reference::operator--()

template<typename Element, typename Pointer, typename Derived>
  typename reference<Element,Pointer,Derived>::derived_type &
    reference<Element,Pointer,Derived>
      ::operator-=(const value_type &rhs)
{
  value_type temp = *this;
  temp -= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference::operator-=()

template<typename Element, typename Pointer, typename Derived>
  typename reference<Element,Pointer,Derived>::derived_type &
    reference<Element,Pointer,Derived>
      ::operator*=(const value_type &rhs)
{
  value_type temp = *this;
  temp *= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference::operator*=()

template<typename Element, typename Pointer, typename Derived>
  typename reference<Element,Pointer,Derived>::derived_type &
    reference<Element,Pointer,Derived>
      ::operator/=(const value_type &rhs)
{
  value_type temp = *this;
  temp /= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference::operator/=()

template<typename Element, typename Pointer, typename Derived>
  typename reference<Element,Pointer,Derived>::derived_type &
    reference<Element,Pointer,Derived>
      ::operator%=(const value_type &rhs)
{
  value_type temp = *this;
  temp %= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference::operator%=()

template<typename Element, typename Pointer, typename Derived>
  typename reference<Element,Pointer,Derived>::derived_type &
    reference<Element,Pointer,Derived>
      ::operator<<=(const value_type &rhs)
{
  value_type temp = *this;
  temp <<= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference::operator<<=()

template<typename Element, typename Pointer, typename Derived>
  typename reference<Element,Pointer,Derived>::derived_type &
    reference<Element,Pointer,Derived>
      ::operator>>=(const value_type &rhs)
{
  value_type temp = *this;
  temp >>= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference::operator>>=()

template<typename Element, typename Pointer, typename Derived>
  typename reference<Element,Pointer,Derived>::derived_type &
    reference<Element,Pointer,Derived>
      ::operator&=(const value_type &rhs)
{
  value_type temp = *this;
  temp &= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference::operator&=()

template<typename Element, typename Pointer, typename Derived>
  typename reference<Element,Pointer,Derived>::derived_type &
    reference<Element,Pointer,Derived>
      ::operator|=(const value_type &rhs)
{
  value_type temp = *this;
  temp |= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference::operator|=()

template<typename Element, typename Pointer, typename Derived>
  typename reference<Element,Pointer,Derived>::derived_type &
    reference<Element,Pointer,Derived>
      ::operator^=(const value_type &rhs)
{
  value_type temp = *this;
  temp ^= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference::operator^=()

  
} // end thrust

