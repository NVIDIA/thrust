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


/*! \file distance.h
 *  \brief Device implementations for distance.
 */

#pragma once

#include <thrust/pair.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/counting_iterator.h>

#include <thrust/inner_product.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace generic
{
namespace detail
{

/////////////
// Structs //
/////////////
//
template<typename T> struct war_nvcc_31_crash { typedef T type; };

// promote small types to int to WAR nvcc 3.1 crash
template<> struct war_nvcc_31_crash<char>           { typedef          int type; };
template<> struct war_nvcc_31_crash<unsigned char>  { typedef unsigned int type; };

template<> struct war_nvcc_31_crash<short>          { typedef          int type; };
template<> struct war_nvcc_31_crash<unsigned short> { typedef unsigned int type; };

template <typename InputType, typename IndexType>
struct element_pair
{
    typename war_nvcc_31_crash<InputType>::type value;
    typename war_nvcc_31_crash<IndexType>::type index;
}; // end element_pair

template <typename InputType, typename IndexType>
struct minmax_element_pair
{
    element_pair<InputType,IndexType> min_pair;
    element_pair<InputType,IndexType> max_pair;
}; // end minmax_element_pair


///////////////
// Functions //
///////////////

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

// return the smaller & larger element making sure to prefer the 
// first occurance of the minimum/maximum element
template <typename InputType, typename IndexType, typename BinaryPredicate>
struct minmax_element_reduction
{
  typedef minmax_element_pair<InputType,IndexType> minmax_pair;
  __host__ __device__ 

  minmax_element_reduction(const BinaryPredicate& _comp) : comp(_comp){}

  __host__ __device__ 
  minmax_element_pair<InputType, IndexType>
  operator()(const minmax_element_pair<InputType, IndexType>& lhs, 
             const minmax_element_pair<InputType, IndexType>& rhs ) const
  {
      minmax_element_pair<InputType,IndexType> result;

      result.min_pair = min_element_reduction<InputType, IndexType, BinaryPredicate>(comp)(lhs.min_pair, rhs.min_pair);
      result.max_pair = max_element_reduction<InputType, IndexType, BinaryPredicate>(comp)(lhs.max_pair, rhs.max_pair);

      return result;
  } // end operator()()

  const BinaryPredicate comp;
}; // end minmax_element_reduction

// binary function that create a element pair (Input,Index)
template <typename InputType, typename IndexType>
struct element_pair_functor
{
    __host__ __device__ 
        element_pair<InputType, IndexType>
        operator()(const InputType& i, const IndexType& n) const 
        {
            return make_element_pair(i, n);
        } // end operator()

}; // end element_pair_functor

// binary function that create a minmax_element pair (Input,Index)
template <typename InputType, typename IndexType>
struct minmax_element_pair_functor
{
    __host__ __device__ 
        minmax_element_pair<InputType, IndexType>
        operator()(const InputType& i, const IndexType& n) const 
        {
            minmax_element_pair<InputType, IndexType> result;
            result.min_pair = make_element_pair(i, n);
            result.max_pair = make_element_pair(i, n);
            return result;
        } // end operator()

}; // end element_pair_functor

} // end detail



template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator min_element(ForwardIterator first, ForwardIterator last,
                          BinaryPredicate comp)
{
    if (first == last)
        return last;

    typedef typename thrust::iterator_traits<ForwardIterator>::value_type      InputType;
    typedef typename thrust::iterator_traits<ForwardIterator>::difference_type IndexType;

    thrust::counting_iterator<IndexType> index_first(0);
    detail::element_pair<InputType, IndexType> init = detail::make_element_pair<InputType, IndexType>(*first, 0);
    detail::min_element_reduction<InputType, IndexType, BinaryPredicate> binary_op1(comp);
    detail::element_pair_functor<InputType, IndexType> binary_op2;

    detail::element_pair<InputType, IndexType> result = thrust::inner_product(first, last, index_first, init, binary_op1, binary_op2);

    return first + result.index;
} // end min_element()

template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator max_element(ForwardIterator first, ForwardIterator last,
                          BinaryPredicate comp)
{
    if (first == last)
        return last;

    typedef typename thrust::iterator_traits<ForwardIterator>::value_type      InputType;
    typedef typename thrust::iterator_traits<ForwardIterator>::difference_type IndexType;

    thrust::counting_iterator<IndexType> index_first(0);
    detail::element_pair<InputType, IndexType> init = detail::make_element_pair<InputType, IndexType>(*first, 0);
    detail::max_element_reduction<InputType, IndexType, BinaryPredicate> binary_op1(comp);
    detail::element_pair_functor<InputType, IndexType> binary_op2;

    detail::element_pair<InputType, IndexType> result = thrust::inner_product(first, last, index_first, init, binary_op1, binary_op2);

    return first + result.index;
} // end max_element()

template <typename ForwardIterator, typename BinaryPredicate>
thrust::pair<ForwardIterator,ForwardIterator> minmax_element(ForwardIterator first, 
                                                             ForwardIterator last,
                                                             BinaryPredicate comp)
{
    if (first == last)
        return thrust::make_pair(last, last);

    typedef typename thrust::iterator_traits<ForwardIterator>::value_type      InputType;
    typedef typename thrust::iterator_traits<ForwardIterator>::difference_type IndexType;

    thrust::counting_iterator<IndexType> index_first(0);

    detail::minmax_element_pair<InputType, IndexType> init;
    init.min_pair = detail::make_element_pair<InputType, IndexType>(*first, 0);
    init.max_pair = detail::make_element_pair<InputType, IndexType>(*first, 0);

    detail::minmax_element_reduction<InputType, IndexType, BinaryPredicate> binary_op1(comp);

    detail::minmax_element_pair_functor<InputType, IndexType> binary_op2;

    detail::minmax_element_pair<InputType, IndexType> result = thrust::inner_product(first, last, index_first, init, binary_op1, binary_op2);

    return thrust::make_pair(first + result.min_pair.index, first + result.max_pair.index);
} // end minmax_element()

} // end namespace generic
} // end namespace device
} // end namespace detail
} // end namespace thrust

