/*
 *  Copyright 2008-2010 NVIDIA Corporation
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


/*! \file scan.h
 *  \brief Defines some functions for computing prefix sums.
 */

#pragma once

#include <thrust/detail/config.h>

namespace thrust
{

/*! \addtogroup algorithms
 */

/*! \addtogroup prefixsums Prefix Sums
 *  \ingroup algorithms
 *  \{
 */

/*! \p inclusive_scan computes an inclusive prefix sum operation. The
 *  term 'inclusive' means that each result includes the corresponding
 *  input operand in the partial sum. More precisely, <tt>*first</tt> is 
 *  assigned to <tt>*result</tt> and the sum of <tt>*first</tt> and 
 *  <tt>*(first + 1)</tt> is assigned to <tt>*(result + 1)</tt>, and so on. 
 *  This version of \p inclusive_scan assumes plus as the associative operator.  
 *  When the input and output sequences are the same, the scan is performed 
 *  in-place.
 
 *  \p inclusive_scan is similar to \c std::partial_sum in the STL.  The primary
 *  difference between the two functions is that \c std::partial_sum guarantees
 *  a serial summation order, while \p inclusive_scan requires associativity of 
 *  the binary operation to parallelize the prefix sum.
 *    
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param result The beginning of the output sequence.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>
 *                        and \c InputIterator's \c value_type is convertible to
 *                        \c OutputIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>,
 *                         and if \c x and \c y are objects of \c OutputIterator's
 *                         \c value_type, then <tt>x + y</tt> is defined. If \c T is
 *                         \c OutputIterator's \c value_type, then <tt>T(0)</tt> is
 *                         defined.
 *
 *  The following code snippet demonstrates how to use \p inclusive_scan
 *
 *  \code
 *  #include <thrust/scan.h>
 *  
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *
 *  thrust::inclusive_scan(data, data + 6, data); // in-place scan
 *
 *  // data is now {1, 1, 3, 5, 6, 9}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/partial_sum.html
 *
 */
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator inclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result);

/*! \p inclusive_scan computes an inclusive prefix sum operation. The
 *  term 'inclusive' means that each result includes the corresponding
 *  input operand in the partial sum.  When the input and output sequences 
 *  are the same, the scan is performed in-place.
 *    
 *  \p inclusive_scan is similar to \c std::partial_sum in the STL.  The primary
 *  difference between the two functions is that \c std::partial_sum guarantees
 *  a serial summation order, while \p inclusive_scan requires associativity of 
 *  the binary operation to parallelize the prefix sum.
 *
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param result The beginning of the output sequence.
 *  \param binary_op The associatve operator used to 'sum' values.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>
 *                        and \c InputIterator's \c value_type is convertible to
 *                        \c OutputIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>
 *                         and \c OutputIterator's \c value_type is convertible to
 *                         both \c AssociativeOperator's \c first_argument_type and
 *                         \c second_argument_type.
 *  \tparam AssociativeOperator is a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>
 *                              and \c AssociativeOperator's \c result_type is
 *                              convertible to \c OutputIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p inclusive_scan
 *
 *  \code
 *  int data[10] = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};
 * 
 *  thrust::maximum<int> binary_op;
 *
 *  thrust::inclusive_scan(data, data + 10, data, 1, binary_op); // in-place scan
 *
 *  // data is now {1, 1, 2, 2, 2, 4, 4, 4, 4, 8}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/partial_sum.html
 */
template<typename InputIterator,
         typename OutputIterator,
         typename AssociativeOperator>
  OutputIterator inclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                AssociativeOperator binary_op);

/*! \p exclusive_scan computes an exclusive prefix sum operation. The
 *  term 'exclusive' means that each result does not include the 
 *  corresponding input operand in the partial sum.  More precisely,
 *  <tt>0</tt> is assigned to <tt>*result</tt> and the sum of 
 *  <tt>0</tt> and <tt>*first</tt> is assigned to <tt>*(result + 1)</tt>,
 *  and so on. This version of \p exclusive_scan assumes plus as the 
 *  associative operator and \c 0 as the initial value.  When the input and 
 *  output sequences are the same, the scan is performed in-place.
 *    
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param result The beginning of the output sequence.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>
 *                        and \c InputIterator's \c value_type is convertible to
 *                        \c OutputIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>,
 *                         and if \c x and \c y are objects of \c OutputIterator's
 *                         \c value_type, then <tt>x + y</tt> is defined. If \c T is
 *                         \c OutputIterator's \c value_type, then <tt>T(0)</tt> is
 *                         defined.
 *
 *  The following code snippet demonstrates how to use \p exclusive_scan
 *
 *  \code
 *  #include <thrust/scan.h>
 *  
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *
 *  thrust::exclusive_scan(data, data + 6, data); // in-place scan
 *
 *  // data is now {0, 1, 1, 3, 5, 6}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/partial_sum.html
 */
template<typename InputIterator,
         typename OutputIterator>
  OutputIterator exclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result);


/*! \p exclusive_scan computes an exclusive prefix sum operation. The
 *  term 'exclusive' means that each result does not include the 
 *  corresponding input operand in the partial sum.  More precisely,
 *  \p init is assigned to <tt>*result</tt> and the sum of \p init and 
 *  <tt>*first</tt> is assigned to <tt>*(result + 1)</tt>, and so on. 
 *  This version of \p exclusive_scan assumes plus as the associative 
 *  operator but requires an initial value \p init.  When the input and 
 *  output sequences are the same, the scan is performed in-place.
 *
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param result The beginning of the output sequence.
 *  \param init The initial value.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>
 *                        and \c InputIterator's \c value_type is convertible to
 *                        \c OutputIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>,
 *                         and if \c x and \c y are objects of \c OutputIterator's
 *                         \c value_type, then <tt>x + y</tt> is defined.
 *  \tparam T is convertible to \c OutputIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p exclusive_scan
 *
 *  \code
 *  #include <thrust/scan.h>
 *  
 *  int data[6] = {1, 0, 2, 2, 1, 3};
 *
 *  thrust::exclusive_scan(data, data + 6, data, 4); // in-place scan
 *
 *  // data is now {4, 5, 5, 7, 9, 10}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/partial_sum.html
 */
template<typename InputIterator,
         typename OutputIterator,
         typename T>
  OutputIterator exclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                const T init);

/*! \p exclusive_scan computes an exclusive prefix sum operation. The
 *  term 'exclusive' means that each result does not include the 
 *  corresponding input operand in the partial sum.  More precisely,
 *  \p init is assigned to <tt>*result</tt> and the value
 *  <tt>binary_op(init, *first)</tt> is assigned to <tt>*(result + 1)</tt>,
 *  and so on. This version of the function requires both and associative 
 *  operator and an initial value \p init.  When the input and output
 *  sequences are the same, the scan is performed in-place.
 *    
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param result The beginning of the output sequence.
 *  \param init The initial value.
 *  \param binary_op The associatve operator used to 'sum' values.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>
 *                        and \c InputIterator's \c value_type is convertible to
 *                        \c OutputIterator's \c value_type.
 *  \tparam OutputIterator is a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>
 *                         and \c OutputIterator's \c value_type is convertible to
 *                         both \c AssociativeOperator's \c first_argument_type and
 *                         \c second_argument_type.
 *  \tparam T is convertible to \c OutputIterator's \c value_type.
 *  \tparam AssociativeOperator is a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>
 *                              and \c AssociativeOperator's \c result_type is
 *                              convertible to \c OutputIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p exclusive_scan
 *
 *  \code
 *  #include <thrust/scan.h>
 *  #include <thrust/functional.h>
 *  
 *  int data[10] = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};
 * 
 *  thrust::maximum<int> binary_op;
 *
 *  thrust::exclusive_scan(data, data + 10, data, 1, binary_op); // in-place scan
 *
 *  // data is now {1, 1, 1, 2, 2, 2, 4, 4, 4, 4 }
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/partial_sum.html
 */
template<typename InputIterator,
         typename OutputIterator,
         typename T,
         typename AssociativeOperator>
  OutputIterator exclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                const T init,
                                AssociativeOperator binary_op);

/*! \} // end prefix sums
 */
	
} // end namespace thrust

#include <thrust/detail/scan.inl>

