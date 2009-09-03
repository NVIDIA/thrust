/*
 *  Copyright 2008-2009 NVIDIA Corporation
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


/*! \file normal_iterator.h
 *  \brief Defines the interface to an iterator class
 *         which adapts a pointer type.
 */

#pragma once

#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>
#include <thrust/device_ptr.h>

namespace thrust
{

namespace detail
{

template<typename Pointer>
  class normal_iterator
    //: public experimental::iterator_adaptor< normal_iterator<Pointer>, Pointer, Pointer >
    : public experimental::iterator_adaptor<
        normal_iterator<Pointer>,
        Pointer,
        Pointer,
        typename thrust::use_default,
        typename thrust::use_default,
        typename thrust::use_default,
        //typename thrust::iterator_traits<Pointer>::reference
        typename thrust::use_default
      >
{
  public:
    __host__ __device__
    normal_iterator() {}

    __host__ __device__
    normal_iterator(Pointer p)
      : normal_iterator::iterator_adaptor_(p) {}
    
    // XXX this needs enable_if_convertible
    //     this is included primarily to create const_iterator from iterator
    template<typename OtherIterator>
    __host__ __device__
    normal_iterator(OtherIterator const & other)
      : normal_iterator::iterator_adaptor_(other.base()) {}

}; // end normal_iterator

template<typename T> struct is_trivial_iterator< normal_iterator<T> > : public true_type {};

namespace device
{

// forward declarations for dereference(device_ptr)
template<typename T>
  inline __device__
    typename iterator_device_reference< device_ptr<T> >::type
      dereference(device_ptr<T> iter);

template<typename T, typename IndexType>
  inline __device__ 
    typename iterator_device_reference< device_ptr<T> >::type
      dereference(device_ptr<T> iter, IndexType n);

template<typename T>
  inline __device__
    typename iterator_device_reference< normal_iterator< device_ptr<T> > >::type
      dereference(normal_iterator< device_ptr<T> > iter)
{
  return dereference(iter.base());
} // end dereference()

template<typename T, typename IndexType>
  inline __device__ 
    typename iterator_device_reference< normal_iterator< device_ptr<T> > >::type
      dereference(normal_iterator< device_ptr<T> > iter, IndexType n)
{
  return dereference(iter.base(), n);
} // end dereference()

} // end device

} // end detail

} // end thrust

