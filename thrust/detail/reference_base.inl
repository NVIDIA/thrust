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

#include <thrust/detail/reference_base.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/memory.h>
#include <thrust/detail/adl_helper.h>

namespace thrust
{
namespace detail
{


template<typename Element, typename Pointer, typename Derived>
  template<typename OtherElement, typename OtherPointer, typename OtherDerived>
    reference_base<Element,Pointer,Derived>
      ::reference_base(const reference_base<OtherElement,OtherPointer,OtherDerived> &other,
                       typename thrust::detail::enable_if_convertible<
                         typename reference_base<OtherElement,OtherPointer,OtherDerived>::pointer,
                         pointer
                       >::type *)
        : m_ptr(other.m_ptr)
{}


template<typename Element, typename Pointer, typename Derived>
  reference_base<Element,Pointer,Derived>
    ::reference_base(const pointer &ptr)
      : m_ptr(ptr)
{}


template<typename Element, typename Pointer, typename Derived>
  typename reference_base<Element,Pointer,Derived>::pointer
    reference_base<Element,Pointer,Derived>
      ::operator&() const
{
  return m_ptr;
} // end reference_base::operator&()


template<typename Element, typename Pointer, typename Derived>
  typename reference_base<Element,Pointer,Derived>::derived_type &
    reference_base<Element,Pointer,Derived>
      ::operator=(const value_type &v)
{
  assign_from(&v);
  return static_cast<derived_type&>(*this);
} // end reference_base::operator=()


template<typename Element, typename Pointer, typename Derived>
  typename reference_base<Element,Pointer,Derived>::derived_type &
    reference_base<Element,Pointer,Derived>
      ::operator=(const reference_base &other)
{
  assign_from(&other); 
  return static_cast<derived_type&>(*this);
} // end reference_base::operator=()


template<typename Element, typename Pointer, typename Derived>
  template<typename OtherElement, typename OtherPointer, typename OtherDerived>
    typename reference_base<Element,Pointer,Derived>::derived_type &
      reference_base<Element,Pointer,Derived>
        ::operator=(const reference_base<OtherElement,OtherPointer,OtherDerived> &other)
{
  assign_from(&other);
  return static_cast<derived_type&>(*this);
} // end reference_base::operator=()


template<typename Element, typename Pointer, typename Derived>
  reference_base<Element,Pointer,Derived>
    ::operator typename reference_base<Element,Pointer,Derived>::value_type () const
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::get_value;

  typedef typename thrust::iterator_space<pointer>::type space;

  return get_value(select_system(space()), m_ptr);
} // end reference_base::operator value_type ()


template<typename Element, typename Pointer, typename Derived>
  template<typename OtherPointer>
    void reference_base<Element,Pointer,Derived>
      ::assign_from(OtherPointer src)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::assign_value;

  typedef typename thrust::iterator_space<pointer>::type      space1;
  typedef typename thrust::iterator_space<OtherPointer>::type space2;

  assign_value(select_system(space1(), space2()), m_ptr, src);
} // end assign_from()


template<typename Element, typename Pointer, typename Derived>
  void reference_base<Element,Pointer,Derived>
    ::swap(derived_type &other)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::iter_swap;

  typedef typename thrust::iterator_space<pointer>::type space;

  iter_swap(select_system(space(), space()), m_ptr, other.m_ptr);
} // end reference_base::swap()


template<typename Element, typename Pointer, typename Derived>
  typename reference_base<Element,Pointer,Derived>::derived_type &
    reference_base<Element,Pointer,Derived>
      ::operator++(void)
{
  value_type temp = *this;
  ++temp;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference_base::operator++()


template<typename Element, typename Pointer, typename Derived>
  typename reference_base<Element,Pointer,Derived>::value_type
    reference_base<Element,Pointer,Derived>
      ::operator++(int)
{
  value_type temp = *this;
  value_type result = temp++;
  *this = temp;
  return result;
} // end reference_base::operator++()


template<typename Element, typename Pointer, typename Derived>
  typename reference_base<Element,Pointer,Derived>::derived_type &
    reference_base<Element,Pointer,Derived>
      ::operator+=(const value_type &rhs)
{
  value_type temp = *this;
  temp += rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference_base::operator+=()

template<typename Element, typename Pointer, typename Derived>
  typename reference_base<Element,Pointer,Derived>::derived_type &
    reference_base<Element,Pointer,Derived>
      ::operator--(void)
{
  value_type temp = *this;
  --temp;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference_base::operator--()

template<typename Element, typename Pointer, typename Derived>
  typename reference_base<Element,Pointer,Derived>::value_type
    reference_base<Element,Pointer,Derived>
      ::operator--(int)
{
  value_type temp = *this;
  value_type result = temp--;
  *this = temp;
  return result;
} // end reference_base::operator--()

template<typename Element, typename Pointer, typename Derived>
  typename reference_base<Element,Pointer,Derived>::derived_type &
    reference_base<Element,Pointer,Derived>
      ::operator-=(const value_type &rhs)
{
  value_type temp = *this;
  temp -= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference_base::operator-=()

template<typename Element, typename Pointer, typename Derived>
  typename reference_base<Element,Pointer,Derived>::derived_type &
    reference_base<Element,Pointer,Derived>
      ::operator*=(const value_type &rhs)
{
  value_type temp = *this;
  temp *= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference_base::operator*=()

template<typename Element, typename Pointer, typename Derived>
  typename reference_base<Element,Pointer,Derived>::derived_type &
    reference_base<Element,Pointer,Derived>
      ::operator/=(const value_type &rhs)
{
  value_type temp = *this;
  temp /= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference_base::operator/=()

template<typename Element, typename Pointer, typename Derived>
  typename reference_base<Element,Pointer,Derived>::derived_type &
    reference_base<Element,Pointer,Derived>
      ::operator%=(const value_type &rhs)
{
  value_type temp = *this;
  temp %= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference_base::operator%=()

template<typename Element, typename Pointer, typename Derived>
  typename reference_base<Element,Pointer,Derived>::derived_type &
    reference_base<Element,Pointer,Derived>
      ::operator<<=(const value_type &rhs)
{
  value_type temp = *this;
  temp <<= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference_base::operator<<=()

template<typename Element, typename Pointer, typename Derived>
  typename reference_base<Element,Pointer,Derived>::derived_type &
    reference_base<Element,Pointer,Derived>
      ::operator>>=(const value_type &rhs)
{
  value_type temp = *this;
  temp >>= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference_base::operator>>=()

template<typename Element, typename Pointer, typename Derived>
  typename reference_base<Element,Pointer,Derived>::derived_type &
    reference_base<Element,Pointer,Derived>
      ::operator&=(const value_type &rhs)
{
  value_type temp = *this;
  temp &= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference_base::operator&=()

template<typename Element, typename Pointer, typename Derived>
  typename reference_base<Element,Pointer,Derived>::derived_type &
    reference_base<Element,Pointer,Derived>
      ::operator|=(const value_type &rhs)
{
  value_type temp = *this;
  temp |= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference_base::operator|=()

template<typename Element, typename Pointer, typename Derived>
  typename reference_base<Element,Pointer,Derived>::derived_type &
    reference_base<Element,Pointer,Derived>
      ::operator^=(const value_type &rhs)
{
  value_type temp = *this;
  temp ^= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference_base::operator^=()

  
} // end detail
} // end thrust

