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

// XXX using adl_helper.h might be too heavy weight for thrust::reference
//     might be a better idea to introduce pointer_adl_helper.h which only
//     makes the pointer interface available to adl
#include <thrust/detail/adl_helper.h>


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
  reference<Element,Pointer,Derived>
    ::operator typename reference<Element,Pointer,Derived>::value_type () const
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::get_value;

  typedef typename thrust::iterator_system<pointer>::type system;

  return get_value(select_system(system()), m_ptr);
} // end reference::operator value_type ()


template<typename Element, typename Pointer, typename Derived>
  template<typename OtherPointer>
    void reference<Element,Pointer,Derived>
      ::assign_from(OtherPointer src)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::assign_value;

  typedef typename thrust::iterator_system<pointer>::type      system1;
  typedef typename thrust::iterator_system<OtherPointer>::type system2;

  assign_value(select_system(system1(), system2()), m_ptr, src);
} // end assign_from()


template<typename Element, typename Pointer, typename Derived>
  void reference<Element,Pointer,Derived>
    ::swap(derived_type &other)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::iter_swap;

  typedef typename thrust::iterator_system<pointer>::type system;

  iter_swap(select_system(system(), system()), m_ptr, other.m_ptr);
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

