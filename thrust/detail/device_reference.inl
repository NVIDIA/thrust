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


/*! \file device_reference.inl
 *  \brief Inline file for device_reference.h.
 */

#include <thrust/device_reference.h>
#include <thrust/copy.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/iterator_traits.h>
#include <iostream>

namespace thrust
{

template<typename T>
  template<typename OtherT>
    device_reference<T>
      ::device_reference(const device_reference<OtherT> &ref

// XXX msvc screws this up
#ifndef _MSC_VER
                         , typename
                         detail::enable_if<
                           detail::is_convertible<
                             typename device_reference<OtherT>::pointer,
                             pointer
                           >::value
                         >::type *dummy
#endif // _MSC_VER
                        )
        :mPtr(ref.mPtr)
{
  ;
} // end device_reference::device_reference()

template<typename T>
  device_reference<T>
    ::device_reference(const pointer &ptr)
      :mPtr(ptr)
{
  ;
} // end device_reference::device_reference()

template<typename T>
  typename device_reference<T>::pointer device_reference<T>
    ::operator&(void) const
{
  return mPtr;
} // end device_reference::operator&()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator=(const T &v)
{
  // test for interoperability
  typedef typename thrust::detail::are_spaces_interoperable<
    typename thrust::iterator_space<pointer>::type,
    thrust::host_space_tag
  >::type interop;

  assign_from(&v, interop());
  return *this;
} // end device_reference::operator=()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator=(const device_reference &ref)
{
  // we're assigning from our own device space
  typedef typename thrust::iterator_space<pointer>::type space;

  // test for interoperability
  typedef typename thrust::detail::are_spaces_interoperable<
    typename thrust::iterator_space<pointer>::type,
    space
  >::type interop;

  assign_from(&ref, interop()); 
  return *this;
} // end device_reference::operator=()

template<typename T>
  template<typename OtherT>
    device_reference<T> &device_reference<T>
      ::operator=(const device_reference<OtherT> &ref)
{
  // we're assigning from an alien space
  typedef typename device_reference<OtherT>::pointer other_pointer;
  typedef typename thrust::iterator_space<other_pointer>::type other_space;

  // test for interoperability
  typedef typename thrust::detail::are_spaces_interoperable<
    typename thrust::iterator_space<pointer>::type,
    other_space
  >::type interop;

  assign_from(&ref, interop());
  return *this;
} // end device_reference::operator=()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator++(void)
{
  value_type temp = *this;
  ++temp;
  *this = temp;
  return *this;
} // end device_reference::operator++()

template<typename T>
  typename device_reference<T>::value_type
    device_reference<T>
      ::operator++(int)
{
  value_type temp = *this;
  value_type result = temp++;
  *this = temp;
  return result;
} // end device_reference::operator++()

template<typename T>
  device_reference<T>
    ::operator typename device_reference<T>::value_type (void) const
{
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
  return *mPtr.get();
#endif
} // end device_reference::operator value_type ()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator+=(const T &rhs)
{
  value_type temp = *this;
  temp += rhs;
  *this = temp;
  return *this;
} // end device_reference::operator+=()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator--(void)
{
  value_type temp = *this;
  --temp;
  *this = temp;
  return *this;
} // end device_reference::operator--()

template<typename T>
  typename device_reference<T>::value_type
    device_reference<T>
      ::operator--(int)
{
  value_type temp = *this;
  value_type result = temp--;
  *this = temp;
  return result;
} // end device_reference::operator--()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator-=(const T &rhs)
{
  value_type temp = *this;
  temp -= rhs;
  *this = temp;
  return *this;
} // end device_reference::operator-=()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator*=(const T &rhs)
{
  value_type temp = *this;
  temp *= rhs;
  *this = temp;
  return *this;
} // end device_reference::operator*=()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator/=(const T &rhs)
{
  value_type temp = *this;
  temp /= rhs;
  *this = temp;
  return *this;
} // end device_reference::operator/=()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator%=(const T &rhs)
{
  value_type temp = *this;
  temp %= rhs;
  *this = temp;
  return *this;
} // end device_reference::operator%=()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator<<=(const T &rhs)
{
  value_type temp = *this;
  temp <<= rhs;
  *this = temp;
  return *this;
} // end device_reference::operator<<=()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator>>=(const T &rhs)
{
  value_type temp = *this;
  temp >>= rhs;
  *this = temp;
  return *this;
} // end device_reference::operator>>=()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator&=(const T &rhs)
{
  value_type temp = *this;
  temp &= rhs;
  *this = temp;
  return *this;
} // end device_reference::operator&=()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator|=(const T &rhs)
{
  value_type temp = *this;
  temp |= rhs;
  *this = temp;
  return *this;
} // end device_reference::operator|=()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator^=(const T &rhs)
{
  value_type temp = *this;
  temp ^= rhs;
  *this = temp;
  return *this;
} // end device_reference::operator^=()

template<typename T>
  template<typename Pointer>
    void device_reference<T>
      ::assign_from(Pointer src, thrust::detail::false_type)
{
  // dispatch copy in general
  thrust::copy(src, src + 1, mPtr);
} // end device_reference::assign_from()

template<typename T>
  template<typename Pointer>
    void device_reference<T>
      ::assign_from(Pointer src, thrust::detail::true_type)
{
  // the spaces are interoperable, just do a simple deref & assign
  *mPtr.get() = *src;
} // end device_reference::assign_from()

template<typename T>
  typename device_reference<T>::value_type
    device_reference<T>
      ::convert(thrust::detail::false_type) const
{
  // dispatch copy in general
  value_type result;
  thrust::copy(mPtr, mPtr + 1, &result);
  return result;
} // end device_reference::convert()

template<typename T>
  typename device_reference<T>::value_type
    device_reference<T>
      ::convert(thrust::detail::true_type) const
{
  // the spaces are interoperable, just do a simple dereference
  return *mPtr.get();
} // end device_reference::convert()

} // end thrust

