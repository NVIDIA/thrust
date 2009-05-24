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


/*! \file reduce.h
 *  \brief Defines the interface to a templated
 *         reduction function.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{

/*! \addtogroup reductions
 *  \{
 */

/*! \p reduce is a generalization of summation: it computes the sum (or some
 *  other binary operation) of all the elements in the range <tt>[first,
 *  last)</tt>. This version of \p reduce uses \c 0 as the initial value of the
 *  reduction. \p reduce is similar to the C++ Standard Template Library's
 *  <tt>std::accumulate</tt>. The primary difference between the two functions
 *  is that <tt>std::accumulate</tt> guarantees the order of summation, while
 *  \p reduce requires associativity of the binary operation to parallelize
 *  the reduction. If the sum operation is not commutative, then
 *  thrust::stable_reduce should be used instead.
 *
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *  \return The result of the reduction.
 *
 *  \tparam InputIterator is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>
 *          and if \c x and \c y are objects of \p InputIterator's \c value_type,
 *          then <tt>x + y</tt> is defined and is convertible to \p InputIterator's
 *          \c value_type. If \c T is \c InputIterator's \c value_type, then
 *          <tt>T(0)</tt> is defined.
 *
 *  The following code snippet demonstrates how to use \p reduce to compute
 *  the sum of a sequence of integers.
 *
 *  \code
 *  #include <thrust/reduce.h>
 *  ...
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *  int result = thrust::reduce(data, data + 6);
 *
 *  // result == 9
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/accumulate.html
 */
template<typename InputIterator> typename
  thrust::iterator_traits<InputIterator>::value_type reduce(InputIterator
      first, InputIterator last);

/*! \p reduce is a generalization of summation: it computes the sum (or some
 *  other binary operation) of all the elements in the range <tt>[first,
 *  last)</tt>. This version of \p reduce uses \p init as the initial value of the
 *  reduction. \p reduce is similar to the C++ Standard Template Library's
 *  <tt>std::accumulate</tt>. The primary difference between the two functions
 *  is that <tt>std::accumulate</tt> guarantees the order of summation, while
 *  \p reduce requires associativity of the binary operation to parallelize
 *  the reduction. If the sum operation is not commutative, then
 *  thrust::stable_reduce should be used instead.
 *
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param init The initial value.
 *  \return The result of the reduction.
 *
 *  \tparam InputIterator is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>
 *          and if \c x and \c y are objects of \p InputIterator's \c value_type,
 *          then <tt>x + y</tt> is defined and is convertible to \p T.
 *  \tparam T is convertible to \p InputIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p reduce to compute
 *  the sum of a sequence of integers including an intialization value.
 *
 *  \code
 *  #include <thrust/scan.h>
 *  ...
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *  int result = thrust::reduce(data, data + 6, 1);
 *
 *  // result == 10
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/accumulate.html
 */
template<typename InputIterator, typename T>
  T reduce(InputIterator first,
           InputIterator last,
           T init);

/*! \p reduce is a generalization of summation: it computes the sum (or some
 *  other binary operation) of all the elements in the range <tt>[first,
 *  last)</tt>. This version of \p reduce uses \p init as the initial value of the
 *  reduction and \p binary_op as the binary function used for summation. \p reduce
 *  is similar to the C++ Standard Template Library's <tt>std::accumulate</tt>.
 *  The primary difference between the two functions is that <tt>std::accumulate</tt>
 *  guarantees the order of summation, while \p reduce requires associativity of
 *  \p binary_op to parallelize the reduction. If \p binary_op is not commutative,
 *  then thrust::stable_reduce should be used instead.
 *
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param init The initial value.
 *  \param binary_op The binary function used to 'sum' values.
 *  \return The result of the reduction.
 *
 *  \tparam InputIterator is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>
 *          and \c InputIterator's \c value_type is convertible to \c OutputIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>
 *          and \c OutputIterator's \c value_type is convertible to both \c AssociativeOperator's \c first_argument_type and
 *          \c second_argument_type.
 *  \tparam T is convertible to \c OutputIterator's \c value_type.
 *  \tparam AssociativeOperator is a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>
 *          and \c AssociativeOperator's \c result_type is convertible to \c OutputIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p reduce to
 *  compute the maximum value of a sequence of integers.
 *
 *  \code
 *  #include <thrust/scan.h>
 *  #include <thrust/functional.h>
 *  ...
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *  int result = thrust::reduce(data, data + 6, -1,
 *                               thrust::maximum<int>());
 *  // result == 3
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/accumulate.html
 */
template<typename InputIterator,
         typename T,
         typename BinaryFunction>
  T reduce(InputIterator first,
           InputIterator last,
           T init,
           BinaryFunction binary_op);

/*! \} // end reductions
 */

} // end namespace thrust

#include <thrust/detail/reduce.inl>

