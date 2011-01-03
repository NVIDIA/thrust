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


/*! \file device_ptr.inl
 *  \brief Inline file for device_ptr.h.
 */

#include <thrust/device_ptr.h>
#include <thrust/device_reference.h>
#include <iostream>

#include <thrust/detail/type_traits.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{

// index operator
template<typename T>
  typename device_ptr<T>::reference device_ptr<T>
    ::operator[](const difference_type &i) const
{
  return reference(device_pointer_cast(mPtr + i));
} // end device_ptr::operator[]()

// dereference operator
template<typename T>
  typename device_ptr<T>::reference device_ptr<T>
    ::operator*(void) const
{
  return reference(device_pointer_cast(mPtr));
} // end device_ptr::operator*()

template<typename T>
  device_ptr<T> device_pointer_cast(T *ptr)
{
  return device_ptr<T>(ptr);
} // end device_pointer_cast()

template<typename T>
  device_ptr<T> device_pointer_cast(const device_ptr<T> &ptr)
{
  return ptr;
} // end device_pointer_cast()

template<typename T>
  T *raw_pointer_cast(const device_ptr<T> &ptr)
{
  return ptr.get();
} // end raw_pointer_cast()

template<typename T>
  T *raw_pointer_cast(T *ptr)
{
  return ptr;
} // end raw_pointer_cast()

// comparison operators follow

// operator==
template<typename T1, typename T2>
  bool operator==(const device_ptr<T1> &lhs, const device_ptr<T2> &rhs)
{
  return lhs.get() == rhs.get();
} // end operator==()

// operator!=
template<typename T1, typename T2>
  bool operator!=(const device_ptr<T1> &lhs, const device_ptr<T2> &rhs)
{
  return !(lhs == rhs);
} // end operator!=()

// operator<
template<typename T1, typename T2>
  bool operator<(const device_ptr<T1> &lhs, const device_ptr<T2> &rhs)
{
  return lhs.get() < rhs.get();
} // end operator<()

// operator<=
template<typename T1, typename T2>
  bool operator<=(const device_ptr<T1> &lhs, const device_ptr<T2> &rhs)
{
  return lhs.get() <= rhs.get();
} // end operator<=()

// operator>
template<typename T1, typename T2>
  bool operator>(const device_ptr<T1> &lhs, const device_ptr<T2> &rhs)
{
  return lhs.get() > rhs.get();
} // end operator>()

// operator>=
template<typename T1, typename T2>
  bool operator>=(const device_ptr<T1> &lhs, const device_ptr<T2> &rhs)
{
  return lhs.get() >= rhs.get();
} // end operator>=()

// output to ostream
template<class E, class T, class Y>
  std::basic_ostream<E, T> &operator<<(std::basic_ostream<E, T> &os, const device_ptr<Y> &p)
{
  return os << p.get();
} // end operator<<()


namespace detail
{

template<typename T>
  struct is_device_ptr< thrust::device_ptr<T> >
    : public true_type
{
}; // end is_device_ptr


namespace device
{


// forward declaration of dereference_result
template<typename T> struct dereference_result;


template<typename T>
  struct dereference_result< device_ptr<T> >
{
  typedef T& type;
}; // end device_traits


template<typename T>
  struct dereference_result< device_ptr<const T> >
{
  typedef const T& type;
}; // end device_traits


template<typename T>
  inline __host__ __device__
    typename dereference_result< device_ptr<T> >::type
      dereference(device_ptr<T> ptr)
{
  return *thrust::raw_pointer_cast(ptr);
} // dereference


template<typename T, typename IndexType>
  inline __host__ __device__
    typename dereference_result< device_ptr<T> >::type
      dereference(thrust::device_ptr<T> ptr, IndexType n)
{
  return thrust::raw_pointer_cast(ptr)[n];
} // dereference

} // end device

} // end namespace detail

} // end namespace thrust

