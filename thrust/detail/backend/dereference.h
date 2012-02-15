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

template<typename Iterator, typename Space>
  class forced_iterator;

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

template<typename Space>
  class discard_iterator;

template <typename IteratorTuple>
  class zip_iterator;

template <typename BidirectionalIterator>
  class reverse_iterator;

template <typename ElementIterator, typename IndexIterator>
  class permutation_iterator;

namespace detail
{

namespace backend
{

template <typename> struct dereference_result {};

// specialize dereference_result for T*
template<typename T>
  struct dereference_result<T*>
{
  typedef T& type;
}; // end dereference_result

// raw pointer
template<typename T>
  inline __host__ __device__
    typename dereference_result<T*>::type
      dereference(T *ptr)
{
  return *ptr;
} // dereference

template<typename T, typename IndexType>
  inline __host__ __device__
    typename dereference_result<T*>::type
      dereference(T *ptr, IndexType n)
{
  return ptr[n];
} // dereference



// device_ptr prototypes
template<typename T>
  inline __host__ __device__
    typename dereference_result< device_ptr<T> >::type
      dereference(device_ptr<T> ptr);

template<typename T, typename IndexType>
  inline __host__ __device__
    typename dereference_result< device_ptr<T> >::type
      dereference(thrust::device_ptr<T> ptr, IndexType n);



// normal_iterator
template<typename Pointer>
  inline __host__ __device__
    typename dereference_result< normal_iterator<Pointer> >::type
      dereference(const normal_iterator<Pointer> &iter);

template<typename Pointer, typename IndexType>
  inline __host__ __device__
    typename dereference_result< normal_iterator<Pointer> >::type
      dereference(const normal_iterator<Pointer> &iter, IndexType n);


// forced_iterator
template<typename Iterator, typename Space>
  inline __host__ __device__
    typename dereference_result< forced_iterator< Iterator, Space > >::type
      dereference(const forced_iterator< Iterator, Space > &iter);

template<typename Iterator, typename Space, typename IndexType>
  inline __host__ __device__
    typename dereference_result< forced_iterator< Iterator, Space > >::type
      dereference(const forced_iterator< Iterator, Space > &iter, IndexType n);



// transform_iterator prototypes
template<typename UnaryFunc, typename Iterator, typename Reference, typename Value>
  inline __host__ __device__
    typename dereference_result< thrust::transform_iterator<UnaryFunc,Iterator,Reference,Value> >::type
      dereference(const thrust::transform_iterator<UnaryFunc,Iterator,Reference,Value> &iter);

template<typename UnaryFunc, typename Iterator, typename Reference, typename Value, typename IndexType>
  inline __host__ __device__
    typename dereference_result< thrust::transform_iterator<UnaryFunc,Iterator,Reference,Value> >::type
      dereference(const thrust::transform_iterator<UnaryFunc,Iterator,Reference,Value> &iter, IndexType n);



// counting_iterator prototypes
template<typename Incrementable, typename Space, typename Traversal, typename Difference>
  inline __host__ __device__
    typename dereference_result< thrust::counting_iterator<Incrementable,Space,Traversal,Difference> >::type
      dereference(const thrust::counting_iterator<Incrementable,Space,Traversal,Difference> &iter);

template<typename Incrementable, typename Space, typename Traversal, typename Difference, typename IndexType>
  inline __host__ __device__
    typename dereference_result< thrust::counting_iterator<Incrementable,Space,Traversal,Difference> >::type
      dereference(const thrust::counting_iterator<Incrementable,Space,Traversal,Difference> &iter, IndexType n);


// constant_iterator prototypes
template<typename Value, typename Incrementable, typename Space>
  inline __host__ __device__
    typename dereference_result< thrust::constant_iterator<Value,Incrementable,Space> >::type
      dereference(const thrust::constant_iterator<Value,Incrementable,Space> &iter);

template<typename Value, typename Incrementable, typename Space, typename IndexType>
  inline __host__ __device__
    typename dereference_result< thrust::constant_iterator<Value,Incrementable,Space> >::type
      dereference(const thrust::constant_iterator<Value,Incrementable,Space> &iter, IndexType n);


// discard_iterator prototypes
template<typename Space>
  inline __host__ __device__
    typename dereference_result< thrust::discard_iterator<Space> >::type
      dereference(const thrust::discard_iterator<Space> &iter);

template<typename Space, typename IndexType>
  inline __host__ __device__
    typename dereference_result< thrust::discard_iterator<Space> >::type
      dereference(const thrust::discard_iterator<Space> &iter, IndexType n);


// zip_iterator prototypes
template<typename IteratorTuple>
  inline __host__ __device__
    typename dereference_result< thrust::zip_iterator<IteratorTuple> >::type
      dereference(const thrust::zip_iterator<IteratorTuple> &iter);

template<typename IteratorTuple, typename IndexType>
  inline __host__ __device__
    typename dereference_result< thrust::zip_iterator<IteratorTuple> >::type
      dereference(const thrust::zip_iterator<IteratorTuple> &iter, IndexType n);


// reverse_iterator prototypes
template<typename BidirectionalIterator>
  inline __host__ __device__
    typename dereference_result< thrust::reverse_iterator<BidirectionalIterator> >::type
      dereference(const thrust::reverse_iterator<BidirectionalIterator> &iter);

template<typename BidirectionalIterator, typename IndexType>
  inline __host__ __device__
    typename dereference_result< thrust::reverse_iterator<BidirectionalIterator> >::type
      dereference(const thrust::reverse_iterator<BidirectionalIterator> &iter, IndexType n);


// permutation_iterator prototypes
template<typename ElementIterator, typename IndexIterator>
  inline __host__ __device__
    typename dereference_result< thrust::permutation_iterator<ElementIterator, IndexIterator> >::type
      dereference(const thrust::permutation_iterator<ElementIterator, IndexIterator> &iter);

template<typename ElementIterator, typename IndexIterator, typename IndexType>
  inline __host__ __device__
    typename dereference_result< thrust::permutation_iterator<ElementIterator, IndexIterator> >::type
      dereference(const thrust::permutation_iterator<ElementIterator, IndexIterator> &iter, IndexType n);

} // end backend

} // end detail

} // end thrust

