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
#include <thrust/detail/backend/generic/select_system.h>
#include <thrust/detail/backend/generic/memory.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/swap.h>
#include <iostream>

#include <thrust/system/cpp/detail/tag.h>

namespace thrust
{
namespace detail
{


template<typename Derived, typename Value, typename Pointer>
  template<typename OtherDerived, typename OtherValue, typename OtherPointer>
    reference_base<Derived,Value,Pointer>
      ::reference_base(const reference_base<OtherDerived,OtherValue,OtherPointer> &other,
                       typename thrust::detail::enable_if_convertible<
                         typename reference_base<OtherDerived,OtherValue,OtherPointer>::pointer,
                         pointer
                       >::type *)
        : m_ptr(other.m_ptr)
{}


template<typename Derived, typename Value, typename Pointer>
  reference_base<Derived,Value,Pointer>
    ::reference_base(const pointer &ptr)
      : m_ptr(ptr)
{}


template<typename Derived, typename Value, typename Pointer>
  typename reference_base<Derived,Value,Pointer>::pointer
    reference_base<Derived,Value,Pointer>
      ::operator&() const
{
  return m_ptr;
} // end reference_base::operator&()


template<typename Derived, typename Value, typename Pointer>
  typename reference_base<Derived,Value,Pointer>::derived_type &
    reference_base<Derived,Value,Pointer>
      ::operator=(const value_type &v)
{
  assign_from(&v);
  return static_cast<derived_type&>(*this);
} // end reference_base::operator=()


template<typename Derived, typename Value, typename Pointer>
  typename reference_base<Derived,Value,Pointer>::derived_type &
    reference_base<Derived,Value,Pointer>
      ::operator=(const reference_base &other)
{
  assign_from(&other); 
  return static_cast<derived_type&>(*this);
} // end reference_base::operator=()


template<typename Derived, typename Value, typename Pointer>
  template<typename OtherDerived, typename OtherValue, typename OtherPointer>
    typename reference_base<Derived,Value,Pointer>::derived_type &
      reference_base<Derived,Value,Pointer>
        ::operator=(const reference_base<OtherDerived,OtherValue,OtherPointer> &other)
{
  assign_from(&other);
  return static_cast<derived_type&>(*this);
} // end reference_base::operator=()


template<typename Derived, typename Value, typename Pointer>
  reference_base<Derived,Value,Pointer>
    ::operator typename reference_base<Derived,Value,Pointer>::value_type () const
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::get_value;

  typedef typename thrust::iterator_space<pointer>::type space;

  return get_value(select_system(space()), m_ptr);
} // end reference_base::operator value_type ()


template<typename Derived,typename Value, typename Pointer>
  template<typename OtherPointer>
    void reference_base<Derived,Value,Pointer>
      ::assign_from(OtherPointer src)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::assign_value;

  typedef typename thrust::iterator_space<pointer>::type      space1;
  typedef typename thrust::iterator_space<OtherPointer>::type space2;

  assign_value(select_system(space1(), space2()), m_ptr, src);
} // end assign_from()


template<typename Derived, typename Value, typename Pointer>
  void reference_base<Derived,Value,Pointer>
    ::swap(derived_type &other)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::iter_swap;

  typedef typename thrust::iterator_space<pointer>::type space;

  iter_swap(select_system(space(), space()), m_ptr, other.m_ptr);
} // end reference_base::swap()


template<typename Derived, typename Value, typename Pointer>
  typename reference_base<Derived,Value,Pointer>::derived_type &
    reference_base<Derived,Value,Pointer>
      ::operator++(void)
{
  value_type temp = *this;
  ++temp;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference_base::operator++()


template<typename Derived, typename Value, typename Pointer>
  typename reference_base<Derived,Value,Pointer>::value_type
    reference_base<Derived,Value,Pointer>
      ::operator++(int)
{
  value_type temp = *this;
  value_type result = temp++;
  *this = temp;
  return result;
} // end reference_base::operator++()


template<typename Derived, typename Value, typename Pointer>
  typename reference_base<Derived,Value,Pointer>::derived_type &
    reference_base<Derived,Value,Pointer>
      ::operator+=(const value_type &rhs)
{
  value_type temp = *this;
  temp += rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference_base::operator+=()

template<typename Derived, typename Value, typename Pointer>
  typename reference_base<Derived,Value,Pointer>::derived_type &
    reference_base<Derived,Value,Pointer>
      ::operator--(void)
{
  value_type temp = *this;
  --temp;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference_base::operator--()

template<typename Derived, typename Value, typename Pointer>
  typename reference_base<Derived,Value,Pointer>::value_type
    reference_base<Derived,Value,Pointer>
      ::operator--(int)
{
  value_type temp = *this;
  value_type result = temp--;
  *this = temp;
  return result;
} // end reference_base::operator--()

template<typename Derived, typename Value, typename Pointer>
  typename reference_base<Derived,Value,Pointer>::derived_type &
    reference_base<Derived,Value,Pointer>
      ::operator-=(const value_type &rhs)
{
  value_type temp = *this;
  temp -= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference_base::operator-=()

template<typename Derived, typename Value, typename Pointer>
  typename reference_base<Derived,Value,Pointer>::derived_type &
    reference_base<Derived,Value,Pointer>
      ::operator*=(const value_type &rhs)
{
  value_type temp = *this;
  temp *= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference_base::operator*=()

template<typename Derived, typename Value, typename Pointer>
  typename reference_base<Derived,Value,Pointer>::derived_type &
    reference_base<Derived,Value,Pointer>
      ::operator/=(const value_type &rhs)
{
  value_type temp = *this;
  temp /= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference_base::operator/=()

template<typename Derived, typename Value, typename Pointer>
  typename reference_base<Derived,Value,Pointer>::derived_type &
    reference_base<Derived,Value,Pointer>
      ::operator%=(const value_type &rhs)
{
  value_type temp = *this;
  temp %= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference_base::operator%=()

template<typename Derived, typename Value, typename Pointer>
  typename reference_base<Derived,Value,Pointer>::derived_type &
    reference_base<Derived,Value,Pointer>
      ::operator<<=(const value_type &rhs)
{
  value_type temp = *this;
  temp <<= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference_base::operator<<=()

template<typename Derived, typename Value, typename Pointer>
  typename reference_base<Derived,Value,Pointer>::derived_type &
    reference_base<Derived,Value,Pointer>
      ::operator>>=(const value_type &rhs)
{
  value_type temp = *this;
  temp >>= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference_base::operator>>=()

template<typename Derived, typename Value, typename Pointer>
  typename reference_base<Derived,Value,Pointer>::derived_type &
    reference_base<Derived,Value,Pointer>
      ::operator&=(const value_type &rhs)
{
  value_type temp = *this;
  temp &= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference_base::operator&=()

template<typename Derived, typename Value, typename Pointer>
  typename reference_base<Derived,Value,Pointer>::derived_type &
    reference_base<Derived,Value,Pointer>
      ::operator|=(const value_type &rhs)
{
  value_type temp = *this;
  temp |= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference_base::operator|=()

template<typename Derived, typename Value, typename Pointer>
  typename reference_base<Derived,Value,Pointer>::derived_type &
    reference_base<Derived,Value,Pointer>
      ::operator^=(const value_type &rhs)
{
  value_type temp = *this;
  temp ^= rhs;
  *this = temp;
  return static_cast<derived_type&>(*this);
} // end reference_base::operator^=()

  
} // end detail
} // end thrust

