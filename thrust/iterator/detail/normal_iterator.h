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


/*! \file normal_iterator.h
 *  \brief Defines the interface to an iterator class
 *         which adapts a pointer type.
 */

#pragma once

#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/backend/dereference.h>
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
    
    template<typename OtherPointer>
    __host__ __device__
    normal_iterator(const normal_iterator<OtherPointer> &other,
                    typename thrust::detail::enable_if_convertible<
                      OtherPointer,
                      Pointer
                    >::type * = 0)
      : normal_iterator::iterator_adaptor_(other.base()) {}

}; // end normal_iterator


template<typename Pointer>
  inline __host__ __device__ normal_iterator<Pointer> make_normal_iterator(Pointer ptr)
{
  return normal_iterator<Pointer>(ptr);
}


template<typename T> struct is_trivial_iterator< normal_iterator<T> > : public true_type {};



namespace backend
{


// specialize dereference_result for normal_iterator with device_ptr as base
template<typename Pointer>
  struct dereference_result< normal_iterator<Pointer> >
{
  typedef typename dereference_result<Pointer>::type type;
}; // end dereference_result


// forward declarations for dereference(device_ptr)
template<typename T>
  inline __host__ __device__
    typename dereference_result< device_ptr<T> >::type
      dereference(device_ptr<T> iter);

template<typename T, typename IndexType>
  inline __host__ __device__
    typename dereference_result< device_ptr<T> >::type
      dereference(device_ptr<T> iter, IndexType n);

// XXX add enable_if<is_convertible<space<Pointer>, host>>
template<typename Pointer>
  inline __host__ __device__
    typename dereference_result< normal_iterator<Pointer> >::type
      dereference(const normal_iterator<Pointer> &iter)
{
  return dereference(iter.base());
} // end dereference()

// XXX add enable_if<is_convertible<space<Pointer>, host>>
template<typename Pointer, typename IndexType>
  inline __host__ __device__
    typename dereference_result< normal_iterator<Pointer> >::type
      dereference(const normal_iterator<Pointer> &iter, IndexType n)
{
  return dereference(iter.base(), n);
} // end dereference()

} // end backend

} // end detail

} // end thrust

