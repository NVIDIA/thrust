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


/*! \file segmented_scan.h
 *  \brief Defines functions for computing segmented prefix-sums.
 */

#pragma once

#include <thrust/detail/config.h>

namespace thrust
{

namespace experimental
{

/*! \addtogroup prefixsums Prefix Sums
 *  \ingroup algorithms
 *  \{
 */

/*! \addtogroup segmentedprefixsums Segmented Prefix Sums
 *  \ingroup prefixsums
 *  \{
 */

/*! \p inclusive_segmented_scan computes an inclusive segmented prefix 
 *  sum operation. The term 'inclusive' means that each result includes 
 *  the corresponding input operand in the partial sum. The term 'segmented'
 *  means that the partial sums are broken into distinct segments.  In other
 *  words, within each segment a separate inclusive scan operation is computed.
 *  Refer to the code sample below for example usage.
 *
 *  This version of \p inclusive_segmented_scan assumes \c plus as the associative
 *  operator used to perform the prefix sum. When the input and output sequences
 *  are the same, the scan is performed in-place.
 * 
 *  This version of \p inclusive_segmented_scan assumes \c equal_to as the binary
 *  predicate used to compare adjacent keys.  Specifically, consecutive iterators
 *  <tt>i</tt> and <tt>i+1</tt> in the range <tt>[first2, first2 + (last1 - first1))</tt>
 *  belong to the same segment if <tt>*i == *(i+1)</tt>, and belong to
 *  different segments otherwise.
 *
 *  \param first1 The beginning of the input sequence.
 *  \param last1 The end of the input sequence.
 *  \param first2 The beginning of the key sequence.
 *  \param result The beginning of the output sequence.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator1 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>
 *                        and \c InputIterator1's \c value_type is convertible to
 *                        \c OutputIterator's \c value_type.
 *  \tparam InputIterator2 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>
 *                        and \c InputIterator2's \c value_type is <a href="http://www.sgi.com/tech/stl/EqualityComparable.html">EqualityComparable</a>.
 *  \tparam OutputIterator is a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>,
 *                         and if \c x and \c y are objects of \c OutputIterator's
 *                         \c value_type, then <tt>x + y</tt> is defined. If \c T is
 *                         \c OutputIterator's \c value_type, then <tt>T(0)</tt> is
 *                         defined.
 *
 *  The following code snippet demonstrates how to use \p inclusive_segmented_scan
 *
 *  \code
 *  #include <thrust/segmented_scan.h>
 *  
 *  int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
 *  int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
 *
 *  thrust::inclusive_segmented_scan(data, data + 6, key, data); // in-place scan
 *
 *  // data is now {1, 2, 3, 1, 2, 1, 1, 2, 3, 4};
 *  \endcode
 *
 *  \see inclusive_scan
 *
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator inclusive_segmented_scan(InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          OutputIterator result);


/*! \p inclusive_segmented_scan computes an inclusive segmented prefix 
 *  sum operation. The term 'inclusive' means that each result includes 
 *  the corresponding input operand in the partial sum. The term 'segmented'
 *  means that the partial sums are broken into distinct segments.  In other
 *  words, within each segment a separate inclusive scan operation is computed.
 *  Refer to the code sample below for example usage.
 *
 *  This version of \p inclusive_segmented_scan uses the associative operator 
 *  \c binary_op to perform the prefix sum. When the input and output sequences
 *  are the same, the scan is performed in-place.
 * 
 *  This version of \p inclusive_segmented_scan assumes \c equal_to as the binary
 *  predicate used to compare adjacent keys.  Specifically, consecutive iterators
 *  <tt>i</tt> and <tt>i+1</tt> in the range <tt>[first2, first2 + (last1 - first1))</tt>
 *  belong to the same segment if <tt>*i == *(i+1)</tt>, and belong to
 *  different segments otherwise.
 *
 *  \param first1 The beginning of the input sequence.
 *  \param last1 The end of the input sequence.
 *  \param first2 The beginning of the key sequence.
 *  \param result The beginning of the output sequence.
 *  \param binary_op The associatve operator used to 'sum' values.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator1 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>
 *                        and \c InputIterator1's \c value_type is convertible to
 *                        \c OutputIterator's \c value_type.
 *  \tparam InputIterator2 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>
 *                        and \c InputIterator2's \c value_type is <a href="http://www.sgi.com/tech/stl/EqualityComparable.html">EqualityComparable</a>.
 *  \tparam OutputIterator is a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>,
 *                         and if \c x and \c y are objects of \c OutputIterator's
 *                         \c value_type, then <tt>x + y</tt> is defined. If \c T is
 *                         \c OutputIterator's \c value_type, then <tt>T(0)</tt> is
 *                         defined.
 *  \tparam AssociativeOperator is a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>
 *                              and \c AssociativeOperator's \c result_type is
 *                              convertible to \c OutputIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p inclusive_segmented_scan
 *
 *  \code
 *  #include <thrust/segmented_scan.h>
 *  #include <thrust/functional.h>
 *  
 *  int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
 *  int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
 *
 *  thrust::inclusive_segmented_scan(data, data + 6, key, data, thrust::plus<int>()); // in-place scan
 *
 *  // data is now {1, 2, 3, 1, 2, 1, 1, 2, 3, 4};
 *  \endcode
 *
 *  \see inclusive_scan
 *
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename AssociativeOperator>
  OutputIterator inclusive_segmented_scan(InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          OutputIterator result,
                                          AssociativeOperator binary_op);


/*! \p inclusive_segmented_scan computes an inclusive segmented prefix 
 *  sum operation. The term 'inclusive' means that each result includes 
 *  the corresponding input operand in the partial sum. The term 'segmented'
 *  means that the partial sums are broken into distinct segments.  In other
 *  words, within each segment a separate inclusive scan operation is computed.
 *  Refer to the code sample below for example usage.
 *
 *  This version of \p inclusive_segmented_scan uses the associative operator 
 *  \c binary_op to perform the prefix sum. When the input and output sequences
 *  are the same, the scan is performed in-place.
 * 
 *  This version of \p inclusive_segmented_scan uses the binary predicate 
 *  \c pred to compare adjacent keys.  Specifically, consecutive iterators
 *  <tt>i</tt> and <tt>i+1</tt> in the range <tt>[first2, first2 + (last1 - first1))</tt>
 *  belong to the same segment if <tt>pred(*i, *(i+1))</tt> is true, and belong to 
 *  different segments otherwise.
 *
 *  \param first1 The beginning of the input sequence.
 *  \param last1 The end of the input sequence.
 *  \param first2 The beginning of the key sequence.
 *  \param result The beginning of the output sequence.
 *  \param binary_op The associatve operator used to 'sum' values.
 *  \param pred  The binary predicate used to determine equality of keys.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator1 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>
 *                        and \c InputIterator1's \c value_type is convertible to
 *                        \c OutputIterator's \c value_type.
 *  \tparam InputIterator2 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>
 *                        and \p ForwardIterator's \c value_type is convertible to \p BinaryPredicate's \c first_argument_type and to \p BinaryPredicate's \c second_argument_type.
 *  \tparam OutputIterator is a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>,
 *                         and if \c x and \c y are objects of \c OutputIterator's
 *                         \c value_type, then <tt>x + y</tt> is defined. If \c T is
 *                         \c OutputIterator's \c value_type, then <tt>T(0)</tt> is
 *                         defined.
 *  \tparam AssociativeOperator is a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>
 *                              and \c AssociativeOperator's \c result_type is
 *                              convertible to \c OutputIterator's \c value_type.
 *  \tparam BinaryPredicate is a model of <a href="http://www.sgi.com/tech/stl/BinaryPredicate.html">Binary Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p inclusive_segmented_scan
 *
 *  \code
 *  #include <thrust/segmented_scan.h>
 *  #include <thrust/functional.h>
 *  
 *  int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
 *  int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
 *
 *  thrust::inclusive_segmented_scan(data, data + 6, key, data, thrust::plus<int>(), thrust::equal_to<int>()); // in-place scan
 *
 *  // data is now {1, 2, 3, 1, 2, 1, 1, 2, 3, 4};
 *  \endcode
 *
 *  \see inclusive_scan
 *
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename AssociativeOperator,
         typename BinaryPredicate>
  OutputIterator inclusive_segmented_scan(InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          OutputIterator result,
                                          AssociativeOperator binary_op,
                                          BinaryPredicate pred);


/*! \p exclusive_segmented_scan computes an exclusive segmented prefix 
 *  sum operation. The term 'exclusive' means that each result does not include
 *  the corresponding input operand in the partial sum. The term 'segmented'
 *  means that the partial sums are broken into distinct segments.  In other
 *  words, within each segment a separate exclusive scan operation is computed.
 *  Refer to the code sample below for example usage.
 *
 *  This version of \p exclusive_segmented_scan assumes uses the value \c 0 to
 *  initialize the exclusive scan operation.
 *
 *  This version of \p exclusive_segmented_scan assumes \c plus as the associative
 *  operator used to perform the prefix sum. When the input and output sequences
 *  are the same, the scan is performed in-place.
 * 
 *  This version of \p exclusive_segmented_scan assumes \c equal_to as the binary
 *  predicate used to compare adjacent keys.  Specifically, consecutive iterators
 *  <tt>i</tt> and <tt>i+1</tt> in the range <tt>[first2, first2 + (last1 - first1))</tt>
 *  belong to the same segment if <tt>*i == *(i+1)</tt>, and belong to 
 *  different segments otherwise.
 *
 *  \param first1 The beginning of the input sequence.
 *  \param last1 The end of the input sequence.
 *  \param first2 The beginning of the key sequence.
 *  \param result The beginning of the output sequence.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator1 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>
 *                        and \c InputIterator1's \c value_type is convertible to
 *                        \c OutputIterator's \c value_type.
 *  \tparam InputIterator2 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>
 *                        and \c InputIterator2's \c value_type is <a href="http://www.sgi.com/tech/stl/EqualityComparable.html">EqualityComparable</a>.
 *  \tparam OutputIterator is a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>,
 *                         and if \c x and \c y are objects of \c OutputIterator's
 *                         \c value_type, then <tt>x + y</tt> is defined. If \c T is
 *                         \c OutputIterator's \c value_type, then <tt>T(0)</tt> is
 *                         defined.
 *
 *  The following code snippet demonstrates how to use \p inclusive_segmented_scan
 *
 *  \code
 *  #include <thrust/segmented_scan.h>
 *  
 *  int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
 *  int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
 *
 *  thrust::exclusive_segmented_scan(data, data + 6, key, data); // in-place scan
 *
 *  // data is now {0, 1, 2, 0, 1, 0, 0, 1, 2, 3};
 *  \endcode
 *
 *  \see exclusive_scan
 *
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator exclusive_segmented_scan(InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          OutputIterator result);


/*! \p exclusive_segmented_scan computes an exclusive segmented prefix 
 *  sum operation. The term 'exclusive' means that each result does not include
 *  the corresponding input operand in the partial sum. The term 'segmented'
 *  means that the partial sums are broken into distinct segments.  In other
 *  words, within each segment a separate exclusive scan operation is computed.
 *  Refer to the code sample below for example usage.
 *
 *  This version of \p exclusive_segmented_scan assumes uses the value \c init to
 *  initialize the exclusive scan operation.
 *
 *  This version of \p exclusive_segmented_scan assumes \c plus as the associative
 *  operator used to perform the prefix sum. When the input and output sequences
 *  are the same, the scan is performed in-place.
 * 
 *  This version of \p exclusive_segmented_scan assumes \c equal_to as the binary
 *  predicate used to compare adjacent keys.  Specifically, consecutive iterators
 *  <tt>i</tt> and <tt>i+1</tt> in the range <tt>[first2, first2 + (last1 - first1))</tt>
 *  belong to the same segment if <tt>*i == *(i+1)</tt>, and belong to 
 *  different segments otherwise.
 *
 *  \param first1 The beginning of the input sequence.
 *  \param last1 The end of the input sequence.
 *  \param first2 The beginning of the key sequence.
 *  \param result The beginning of the output sequence.
 *  \param init The initial value.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator1 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>
 *                        and \c InputIterator1's \c value_type is convertible to
 *                        \c OutputIterator's \c value_type.
 *  \tparam InputIterator2 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>
 *                        and \c InputIterator2's \c value_type is <a href="http://www.sgi.com/tech/stl/EqualityComparable.html">EqualityComparable</a>.
 *  \tparam OutputIterator is a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>,
 *                         and if \c x and \c y are objects of \c OutputIterator's
 *                         \c value_type, then <tt>x + y</tt> is defined. If \c T is
 *                         \c OutputIterator's \c value_type, then <tt>T(0)</tt> is
 *                         defined.
 *  \tparam T is convertible to \c OutputIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p inclusive_segmented_scan
 *
 *  \code
 *  #include <thrust/segmented_scan.h>
 *  
 *  int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
 *  int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
 *
 *  thrust::exclusive_segmented_scan(data, data + 6, key, data, 5); // in-place scan
 *
 *  // data is now {5, 6, 7, 5, 6, 5, 5, 6, 7, 8};
 *  \endcode
 *
 *  \see exclusive_scan
 *
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T>
  OutputIterator exclusive_segmented_scan(InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          OutputIterator result,
                                          const T init);


/*! \p exclusive_segmented_scan computes an exclusive segmented prefix 
 *  sum operation. The term 'exclusive' means that each result does not include
 *  the corresponding input operand in the partial sum. The term 'segmented'
 *  means that the partial sums are broken into distinct segments.  In other
 *  words, within each segment a separate exclusive scan operation is computed.
 *  Refer to the code sample below for example usage.
 *
 *  This version of \p exclusive_segmented_scan assumes uses the value \c init to
 *  initialize the exclusive scan operation.
 *
 *  This version of \p exclusive_segmented_scan uses the associative operator 
 *  \c binary_op to perform the prefix sum. When the input and output sequences
 *  are the same, the scan is performed in-place.
 * 
 *  This version of \p exclusive_segmented_scan assumes \c equal_to as the binary
 *  predicate used to compare adjacent keys.  Specifically, consecutive iterators
 *  <tt>i</tt> and <tt>i+1</tt> in the range <tt>[first2, first2 + (last1 - first1))</tt>
 *  belong to the same segment if <tt>*i == *(i+1)</tt>, and belong to
 *  different segments otherwise.
 *
 *  \param first1 The beginning of the input sequence.
 *  \param last1 The end of the input sequence.
 *  \param first2 The beginning of the key sequence.
 *  \param result The beginning of the output sequence.
 *  \param init The initial value.
 *  \param binary_op The associatve operator used to 'sum' values.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator1 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>
 *                        and \c InputIterator1's \c value_type is convertible to
 *                        \c OutputIterator's \c value_type.
 *  \tparam InputIterator2 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>
 *                        and \c InputIterator2's \c value_type is <a href="http://www.sgi.com/tech/stl/EqualityComparable.html">EqualityComparable</a>.
 *  \tparam OutputIterator is a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>,
 *                         and if \c x and \c y are objects of \c OutputIterator's
 *                         \c value_type, then <tt>x + y</tt> is defined. If \c T is
 *                         \c OutputIterator's \c value_type, then <tt>T(0)</tt> is
 *                         defined.
 *  \tparam T is convertible to \c OutputIterator's \c value_type.
 *  \tparam AssociativeOperator is a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>
 *                              and \c AssociativeOperator's \c result_type is
 *                              convertible to \c OutputIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p inclusive_segmented_scan
 *
 *  \code
 *  #include <thrust/segmented_scan.h>
 *  #include <thrust/functional.h>
 *  
 *  int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
 *  int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
 *
 *  thrust::exclusive_segmented_scan(data, data + 6, key, data, 5, thrust::plus<int>()); // in-place scan
 *
 *  // data is now {5, 6, 7, 5, 6, 5, 5, 6, 7, 8};
 *  \endcode
 *
 *  \see exclusive_scan
 *
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename AssociativeOperator>
  OutputIterator exclusive_segmented_scan(InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          OutputIterator result,
                                          const T init,
                                          AssociativeOperator binary_op);


/*! \p exclusive_segmented_scan computes an exclusive segmented prefix 
 *  sum operation. The term 'exclusive' means that each result does not include
 *  the corresponding input operand in the partial sum. The term 'segmented'
 *  means that the partial sums are broken into distinct segments.  In other
 *  words, within each segment a separate exclusive scan operation is computed.
 *  Refer to the code sample below for example usage.
 *
 *  This version of \p exclusive_segmented_scan assumes uses the value \c init to
 *  initialize the exclusive scan operation.
 *
 *  This version of \p exclusive_segmented_scan uses the associative operator 
 *  \c binary_op to perform the prefix sum. When the input and output sequences
 *  are the same, the scan is performed in-place.
 * 
 *  This version of \p exclusive_segmented_scan uses the binary predicate \c pred
 *  to compare adjacent keys.  Specifically, consecutive iterators
 *  <tt>i</tt> and <tt>i+1</tt> in the range <tt>[first2, first2 + (last1 - first1))</tt>
 *  belong to the same segment if <tt>pred(*i, *(i+1))</tt> is true, and belong to 
 *  different segments otherwise.
 *
 *  \param first1 The beginning of the input sequence.
 *  \param last1 The end of the input sequence.
 *  \param first2 The beginning of the key sequence.
 *  \param result The beginning of the output sequence.
 *  \param init The initial value.
 *  \param binary_op The associatve operator used to 'sum' values.
 *  \param pred  The binary predicate used to determine equality of keys.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator1 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>
 *                        and \c InputIterator1's \c value_type is convertible to
 *                        \c OutputIterator's \c value_type.
 *  \tparam InputIterator2 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>
 *                        and \c InputIterator2's \c value_type is <a href="http://www.sgi.com/tech/stl/EqualityComparable.html">EqualityComparable</a>.
 *  \tparam OutputIterator is a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>,
 *                         and if \c x and \c y are objects of \c OutputIterator's
 *                         \c value_type, then <tt>x + y</tt> is defined. If \c T is
 *                         \c OutputIterator's \c value_type, then <tt>T(0)</tt> is
 *                         defined.
 *  \tparam T is convertible to \c OutputIterator's \c value_type.
 *  \tparam AssociativeOperator is a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>
 *                              and \c AssociativeOperator's \c result_type is
 *                              convertible to \c OutputIterator's \c value_type.
 *  \tparam BinaryPredicate is a model of <a href="http://www.sgi.com/tech/stl/BinaryPredicate.html">Binary Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p inclusive_segmented_scan
 *
 *  \code
 *  #include <thrust/segmented_scan.h>
 *  #include <thrust/functional.h>
 *  
 *  int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
 *  int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
 *
 *  thrust::exclusive_segmented_scan(data, data + 6, key, data, 5, thrust::plus<int>(), thrust::equal_to<int>()); // in-place scan
 *
 *  // data is now {5, 6, 7, 5, 6, 5, 5, 6, 7, 8};
 *  \endcode
 *
 *  \see exclusive_scan
 *
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename AssociativeOperator,
         typename BinaryPredicate>
  OutputIterator exclusive_segmented_scan(InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          OutputIterator result,
                                          const T init,
                                          AssociativeOperator binary_op,
                                          BinaryPredicate pred);

/*! \} // end segmentedprefixsums
 */

/*! \} // end prefix sums
 */

} // end namespace experimental
	
} // end namespace thrust

#include <thrust/detail/segmented_scan.inl>

