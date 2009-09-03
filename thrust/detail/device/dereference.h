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


/*! \file dereference.h
 *  \brief Overloads for everything dereferenceable by dereference.h
 */

#pragma once

#include <thrust/detail/type_traits.h>

namespace thrust
{

// forward declarations of iterator types
template<typename T>
  class device_ptr;

namespace detail
{

template<typename Pointer>
  class normal_iterator;

} // end detail


template<typename Value,
         typename Incrementable,
         typename Space>
  class constant_iterator;


template <class UnaryFunc, class Iterator, class Reference, class Value>
  class transform_iterator;

template<typename Incrementable,
         typename Space,
         typename Traversal,
         typename Difference>
  class counting_iterator;

template <typename IteratorTuple>
  class zip_iterator;

namespace detail
{

namespace device
{

// raw pointer
template<typename T>
  inline __device__
    typename iterator_device_reference<T*>::type
      dereference(T *ptr)
{
  return *ptr;
} // dereference

template<typename T, typename IndexType>
  inline __device__
    typename iterator_device_reference<T*>::type
      dereference(T *ptr, IndexType n)
{
  return ptr[n];
} // dereference



// device_ptr prototypes
template<typename T>
  inline __device__
    typename iterator_device_reference< device_ptr<T> >::type
      dereference(device_ptr<T> ptr);

template<typename T, typename IndexType>
  inline __device__
    typename iterator_device_reference< device_ptr<T> >::type
      dereference(thrust::device_ptr<T> ptr, IndexType n);



// normal_iterator
template<typename T>
  inline __device__
    typename iterator_device_reference< normal_iterator< device_ptr<T> > >::type
      dereference(normal_iterator< device_ptr<T> > iter);

template<typename T, typename IndexType>
  inline __device__ 
    typename iterator_device_reference< normal_iterator< device_ptr<T> > >::type
      dereference(normal_iterator< device_ptr<T> > iter, IndexType n);



// transform_iterator prototypes
template<typename UnaryFunc, typename Iterator, typename Reference, typename Value>
  inline __device__
    typename iterator_device_reference< thrust::transform_iterator<UnaryFunc,Iterator,Reference,Value> >::type
      dereference(thrust::transform_iterator<UnaryFunc,Iterator,Reference,Value> iter);

template<typename UnaryFunc, typename Iterator, typename Reference, typename Value, typename IndexType>
  inline __device__
    typename iterator_device_reference< thrust::transform_iterator<UnaryFunc,Iterator,Reference,Value> >::type
      dereference(thrust::transform_iterator<UnaryFunc,Iterator,Reference,Value> iter, IndexType n);



// counting_iterator prototypes
template<typename Incrementable, typename Space, typename Traversal, typename Difference>
  inline __device__
    typename iterator_device_reference< thrust::counting_iterator<Incrementable,Space,Traversal,Difference> >::type
      dereference(thrust::counting_iterator<Incrementable,Space,Traversal,Difference> iter);

template<typename Incrementable, typename Space, typename Traversal, typename Difference, typename IndexType>
  inline __device__
    typename iterator_device_reference< thrust::counting_iterator<Incrementable,Space,Traversal,Difference> >::type
      dereference(thrust::counting_iterator<Incrementable,Space,Traversal,Difference> iter, IndexType n);


// constant_iterator prototypes
template<typename Value, typename Incrementable, typename Space>
  inline __device__
    typename iterator_device_reference< thrust::constant_iterator<Value,Incrementable,Space> >::type
      dereference(thrust::constant_iterator<Value,Incrementable,Space> iter);

template<typename Value, typename Incrementable, typename Space, typename IndexType>
  inline __device__
    typename iterator_device_reference< thrust::constant_iterator<Value,Incrementable,Space> >::type
      dereference(thrust::constant_iterator<Value,Incrementable,Space> iter, IndexType n);


// zip_iterator prototypes
template<typename IteratorTuple>
  inline __device__
    typename iterator_device_reference< thrust::zip_iterator<IteratorTuple> >::type
      dereference(thrust::zip_iterator<IteratorTuple> iter);

template<typename IteratorTuple, typename IndexType>
  inline __device__
    typename iterator_device_reference< thrust::zip_iterator<IteratorTuple> >::type
      dereference(thrust::zip_iterator<IteratorTuple> iter, IndexType n);

} // end device

} // end detail

} // end thrust

