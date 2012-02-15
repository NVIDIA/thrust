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

#include <thrust/detail/reference_base.h>
#include <thrust/detail/copy.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/swap.h>
#include <iostream>

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
  // test for interoperability
  typedef typename thrust::detail::are_spaces_interoperable<
    typename thrust::iterator_space<pointer>::type,
    thrust::host_space_tag
  >::type interop;

  assign_from(&v, interop());

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
// XXX we should only do this check in cuda::reference
#ifndef __CUDA_ARCH__
  // get our device space
  typedef typename thrust::iterator_space<pointer>::type space;

  // test for interoperability with host_space
  typedef typename thrust::detail::are_spaces_interoperable<
    typename thrust::iterator_space<pointer>::type,
    host_space_tag
  >::type interop;

  return convert(interop());
#else
  return *m_ptr.get();
#endif
} // end reference_base::operator value_type ()


template<typename Derived,typename Value, typename Pointer>
  template<typename OtherPointer>
    void reference_base<Derived,Value,Pointer>
      ::assign_from(OtherPointer src)
{
  // test for interoperability between three spaces:
  // 1. the other reference's space
  // 2. this reference's space
  // 3. the space of the calling function
  typedef typename thrust::iterator_space<OtherPointer>::type other_space;
  typedef typename thrust::iterator_space<pointer>::type      this_space;

  // XXX this could potentially be something other than host
  typedef thrust::host_space_tag caller_space;

  // test for interoperability between this and other
  typedef typename thrust::detail::are_spaces_interoperable<
    this_space,
    other_space
  >::type interop1;

  // test for interoperability between caller and this
  typedef typename thrust::detail::are_spaces_interoperable<
    caller_space,
    this_space
  >::type interop2;

  // test for interoperability between caller and other
  typedef typename thrust::detail::are_spaces_interoperable<
    caller_space,
    this_space
  >::type interop3;

  // we require interoperability of everything
  typedef thrust::detail::and_<
    interop1,
    interop2,
    interop3
  > interop;

  assign_from(src, interop());
} // end assign_from()


template<typename Derived, typename Value, typename Pointer>
  template<typename OtherPointer>
    void reference_base<Derived,Value,Pointer>
      ::assign_from(OtherPointer src, thrust::detail::false_type)
{
  // dispatch copy in general
  thrust::copy(src, src + 1, m_ptr);
} // end reference_base::assign_from()


template<typename Derived, typename Value, typename Pointer>
  template<typename OtherPointer>
    void reference_base<Derived,Value,Pointer>
      ::assign_from(OtherPointer src, thrust::detail::true_type)
{
  // the spaces are interoperable, just do a simple deref & assign
  *m_ptr.get() = *src;
} // end reference_base::assign_from()


template<typename Derived, typename Value, typename Pointer>
  typename reference_base<Derived,Value,Pointer>::value_type
    reference_base<Derived,Value,Pointer>
      ::convert(thrust::detail::false_type) const
{
  // dispatch copy in general
  value_type result = value_type();
  thrust::copy(m_ptr, m_ptr + 1, &result);
  return result;
} // end reference_base::convert()


template<typename Derived, typename Value, typename Pointer>
  typename reference_base<Derived,Value,Pointer>::value_type
    reference_base<Derived,Value,Pointer>
      ::convert(thrust::detail::true_type) const
{
  // the spaces are interoperable, just do a simple dereference
  return *m_ptr.get();
} // end reference_base::convert()


template<typename Derived, typename Value, typename Pointer>
  void reference_base<Derived,Value,Pointer>
    ::swap(derived_type &other)
{
#ifndef __CUDA_ARCH__
  // get our device space
  typedef typename thrust::iterator_space<pointer>::type space;

  // test for interoperability with host_space
  typedef typename thrust::detail::are_spaces_interoperable<
    typename thrust::iterator_space<pointer>::type,
    host_space_tag
  >::type interop;

  swap(other, interop());
#else
  // use unqualified swap to ensure that user-defined swap gets caught by ADL
  using thrust::swap;
  swap(*m_ptr.get(), *other.m_ptr.get());
#endif
} // end reference_base::swap()


template<typename Derived, typename Value, typename Pointer>
  void reference_base<Derived,Value,Pointer>
    ::swap(reference_base<Derived,Value,Pointer> &other,
           thrust::detail::true_type)
{
  // the spaces are interoperable, just do a simple deref & swap
  // use unqualified swap to ensure that user-defined swap gets caught by ADL
  using thrust::swap;
  swap(*m_ptr.get(), *other.m_ptr.get());
} // end reference_basee::swap()


template<typename Derived, typename Value, typename Pointer>
  void reference_base<Derived,Value,Pointer>
    ::swap(reference_base<Derived,Value,Pointer> &other,
           thrust::detail::false_type)
{
  // dispatch swap_ranges
  thrust::swap_ranges(m_ptr, m_ptr + 1, other.m_ptr);
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

