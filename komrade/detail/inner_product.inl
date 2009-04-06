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


/*! \file inner_product.inl
 *  \brief Inline file for inner_product.h.
 */

#include <komrade/inner_product.h>
#include <komrade/functional.h>
#include <komrade/iterator/iterator_traits.h>
#include <komrade/detail/dispatch/inner_product.h>

namespace komrade
{


// the standard mathematical inner_product with plus and multiplies
template <typename InputIterator1, typename InputIterator2, typename OutputType>
OutputType 
inner_product(InputIterator1 first1, InputIterator1 last1,
              InputIterator2 first2, OutputType init)
{
  komrade::plus<OutputType>       binary_op1;
  komrade::multiplies<OutputType> binary_op2;
  return komrade::inner_product(first1, last1, first2, init, binary_op1, binary_op2);
} // end inner_product()

// the generalized inner_product with two binary functions
template <typename InputIterator1, typename InputIterator2, typename OutputType,
          typename BinaryFunction1, typename BinaryFunction2>
OutputType
inner_product(InputIterator1 first1, InputIterator1 last1,
              InputIterator2 first2, OutputType init, 
              BinaryFunction1 binary_op1, BinaryFunction2 binary_op2)
{
  // dispatch on category
  return komrade::detail::dispatch::inner_product(first1, last1, first2, init, binary_op1, binary_op2,
    typename komrade::iterator_traits<InputIterator1>::iterator_category(),
    typename komrade::iterator_traits<InputIterator2>::iterator_category());
} // end inner_product()


} // end namespace komrade

