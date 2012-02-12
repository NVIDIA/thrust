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


/*! \file inner_product.inl
 *  \brief Inline file for inner_product.h.
 */

#include <thrust/generate.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/inner_product.h>
#include <thrust/detail/adl_helper.h>

namespace thrust
{


template <typename InputIterator1, typename InputIterator2, typename OutputType>
OutputType 
inner_product(InputIterator1 first1, InputIterator1 last1,
              InputIterator2 first2, OutputType init)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::inner_product;

  typedef typename thrust::iterator_system<InputIterator1>::type system1;
  typedef typename thrust::iterator_system<InputIterator2>::type system2;

  return inner_product(select_system(system1(),system2()), first1, last1, first2, init);
} // end inner_product()

template <typename InputIterator1, typename InputIterator2, typename OutputType,
          typename BinaryFunction1, typename BinaryFunction2>
OutputType
inner_product(InputIterator1 first1, InputIterator1 last1,
              InputIterator2 first2, OutputType init, 
              BinaryFunction1 binary_op1, BinaryFunction2 binary_op2)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::inner_product;

  typedef typename thrust::iterator_system<InputIterator1>::type system1;
  typedef typename thrust::iterator_system<InputIterator2>::type system2;

  return inner_product(select_system(system1(),system2()), first1, last1, first2, init, binary_op1, binary_op2);
} // end inner_product()


} // end namespace thrust

