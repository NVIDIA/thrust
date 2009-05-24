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


/*! \file min_max_element.h
 *  \brief Defines the interface to
 *         the dispatch layers of the
 *         min_element and max_element functions.
 */

#pragma once

#include <algorithm>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/device/cuda/reduce.h>

namespace thrust
{

namespace detail
{

namespace dispatch
{

namespace detail
{


/////////////
// Structs //
/////////////


template <typename InputType, typename IndexType>
struct element_pair
{
    InputType value;
    IndexType index;
}; // end element_pair


template <typename InputType, typename IndexType>
__host__ __device__
element_pair<InputType,IndexType> make_element_pair(const InputType &value,
                                                    const IndexType &index)
{
  element_pair<InputType,IndexType> result;
  result.value = value;
  result.index = index;
  return result;
} // end make_element_pair()


//////////////
// Functors //
//////////////


// return the smaller/larger element making sure to prefer the 
// first occurance of the minimum/maximum element
template <typename InputType, typename IndexType, typename BinaryPredicate>
struct min_element_reduction
{
  typedef element_pair<InputType,IndexType> min_pair;
  __host__ __device__ 

  min_element_reduction(const BinaryPredicate& _comp) : comp(_comp){}

  __host__ __device__ 
  element_pair<InputType, IndexType>
  operator()(const element_pair<InputType, IndexType>& lhs, 
             const element_pair<InputType, IndexType>& rhs ) const
  {
    if(comp(lhs.value,rhs.value))
      return lhs;
    if(comp(rhs.value,lhs.value))
      return rhs;

    if(lhs.index < rhs.index)
      return lhs;
    else
      return rhs;
  } // end operator()()

  const BinaryPredicate comp;
}; // end min_element_reduction


template <typename InputType, typename IndexType, typename BinaryPredicate>
struct max_element_reduction
{
  typedef element_pair<InputType,IndexType> max_pair;

  __host__ __device__ 
  max_element_reduction(const BinaryPredicate& _comp) : comp(_comp){}

  __host__ __device__ 
  element_pair<InputType, IndexType>
  operator()(const element_pair<InputType, IndexType>& lhs, 
             const element_pair<InputType, IndexType>& rhs ) const
  {
    if(comp(lhs.value, rhs.value))
      return rhs;
    if(comp(rhs.value, lhs.value))
      return lhs;

    if(lhs.index < rhs.index)
      return lhs;
    else
      return rhs;
  } // end operator()()

  const BinaryPredicate comp;
}; // end max_element_reduction


// given an integer n produce the tuple (array[n], n)
template <typename InputType, typename IndexType>
struct element_functor
{
  __host__ __device__ 
  element_functor(const InputType * _array) : array(_array){}

  __host__ __device__ 
  element_pair<InputType, IndexType>
  operator[](const IndexType& n) const 
  {
    return make_element_pair(array[n], n);
  } // end operator[]()

  const InputType * array;
}; // end element_functor

} // end detail

////////////////
// Host Paths //
////////////////
template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator min_element(ForwardIterator first, 
                            ForwardIterator last,
                            BinaryPredicate comp, 
                            thrust::input_host_iterator_tag)
{
  return std::min_element(first, last, comp);
} // end min_element()


template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator max_element(ForwardIterator first, 
                            ForwardIterator last,
                            BinaryPredicate comp, 
                            thrust::input_host_iterator_tag)
{
  return std::max_element(first, last, comp);
} // end max_element()


//////////////////
// Device Paths //
//////////////////
template <typename InputIterator, typename BinaryPredicate>
InputIterator min_element(InputIterator first, InputIterator last,
                          BinaryPredicate comp, 
                          thrust::random_access_device_iterator_tag)
{
  typedef typename thrust::iterator_traits<InputIterator>::value_type      InputType;
  typedef typename thrust::iterator_traits<InputIterator>::difference_type IndexType;
  
  const IndexType n = last - first;

  // empty range
  if(n == 0)
    return last;

  detail::element_pair<InputType, IndexType> init = detail::make_element_pair<InputType,IndexType>(*first,0);
  // XXX use make_device_dereferenceable here instead of assuming &*first is device_ptr
  detail::element_functor<InputType,IndexType> F((&*first).get());
  detail::min_element_reduction<InputType, IndexType, BinaryPredicate> binary_op(comp);
  
  detail::element_pair<InputType, IndexType> result = thrust::detail::device::cuda::reduce(F, n, init, binary_op);

  return first + result.index;
} // end min_element()

template <typename InputIterator, typename BinaryPredicate>
InputIterator max_element(InputIterator first, InputIterator last,
                          BinaryPredicate comp, 
                          thrust::random_access_device_iterator_tag)
{
  typedef typename thrust::iterator_traits<InputIterator>::value_type      InputType;
  typedef typename thrust::iterator_traits<InputIterator>::difference_type IndexType;
  
  const IndexType n = last - first;

  // empty range
  if(n == 0)
    return last;

  detail::element_pair<InputType, IndexType> init = detail::make_element_pair<InputType, IndexType>(*first,0);
  // XXX use make_device_dereferenceable here instead of assuming &*first is device_ptr
  detail::element_functor<InputType,IndexType> F((&*first).get());
  detail::max_element_reduction<InputType, IndexType, BinaryPredicate> binary_op(comp);
  
  detail::element_pair<InputType, IndexType> result = thrust::detail::device::cuda::reduce(F, n, init, binary_op);

  return first + result.index;
} // end max_element()

} // end dispatch

} // end detail

} // end thrust

