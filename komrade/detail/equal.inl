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

#include <komrade/equal.h>
#include <komrade/functional.h>
#include <komrade/inner_product.h>

namespace komrade
{

namespace detail
{

// this differs from komrade::equal_to in that the 
// types of the operands may differ
template <typename T1, typename T2>
struct operator_equal
{
  __host__ __device__
  bool operator()(const T1 &lhs, const T2 &rhs) const 
  {
    //TODO, should this return an int type for speed?
    return lhs == rhs;
  } // end operator()()
}; // end operator_equal

} // end detail


template <typename InputIterator1, typename InputIterator2>
bool equal(InputIterator1 first1, InputIterator1 last1,
           InputIterator2 first2)
{
  typedef typename komrade::iterator_traits<InputIterator1>::value_type InputType1;
  typedef typename komrade::iterator_traits<InputIterator2>::value_type InputType2;

  komrade::detail::operator_equal<InputType1, InputType2> eq;
  return komrade::equal(first1, last1, first2, eq);
} // end equal()


template <typename InputIterator1, typename InputIterator2, 
          typename BinaryPredicate>
bool equal(InputIterator1 first1, InputIterator1 last1,
           InputIterator2 first2, BinaryPredicate binary_pred)
{
  komrade::logical_and<int> binary_op1; // the "plus" of the inner_product
  return static_cast<bool>( komrade::inner_product(first1, last1, first2, static_cast<int>(1), binary_op1, binary_pred) );
} // end equal()


} // end komrade

