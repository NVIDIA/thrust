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


/*! \file equal.inl
 *  \brief Inline file for equal.h.
 */

#include <thrust/functional.h>
#include <thrust/detail/dispatch/equal.h>

namespace thrust
{

namespace detail
{

// this differs from thrust::equal_to in that the 
// types of the operands may differ
template <typename T1, typename T2>
struct operator_equal
{
  __host__ __device__
  bool operator()(const T1 &lhs, const T2 &rhs) const 
  {
      return lhs == rhs;
  } // end operator()()
}; // end operator_equal

} // end detail


template <typename InputIterator1, typename InputIterator2>
bool equal(InputIterator1 first1, InputIterator1 last1,
           InputIterator2 first2)
{
  typedef typename thrust::iterator_traits<InputIterator1>::value_type InputType1;
  typedef typename thrust::iterator_traits<InputIterator2>::value_type InputType2;

  thrust::detail::operator_equal<InputType1, InputType2> eq;
  return thrust::equal(first1, last1, first2, eq);
} // end equal()


template <typename InputIterator1, typename InputIterator2, 
          typename BinaryPredicate>
bool equal(InputIterator1 first1, InputIterator1 last1,
           InputIterator2 first2, BinaryPredicate binary_pred)
{
  return thrust::detail::dispatch::equal(first1, last1, first2, binary_pred,
          typename thrust::iterator_space<InputIterator1>::type(),
          typename thrust::iterator_space<InputIterator2>::type());
} // end equal()


} // end thrust

