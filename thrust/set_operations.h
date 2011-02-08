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


/*! \file set_operations.h
 *  \brief Set operations for sorted ranges.
 */

#pragma once

#include <thrust/detail/config.h>

namespace thrust
{

/*! \addtogroup set_operations Set Operations
 *  \ingroup algorithms
 *  \{
 */

/*! \p set_difference constructs a sorted range that is the set difference of the sorted
 *  ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>. The return value is the
 *  end of the output range.
 *
 *  In the simplest case, \p set_difference performs the "difference" operation from set
 *  theory: the output range contains a copy of every element that is contained in
 *  <tt>[first1, last1)</tt> and not contained in <tt>[first2, last1)</tt>. The general case
 *  is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[first1, last1)</tt> contains \c m elements
 *  that are equivalent to each other and if <tt>[first2, last2)</tt> contains \c n
 *  elements that are equivalent to them, the last <tt>max(m-n,0)</tt> elements from
 *  <tt>[first1, last1)</tt> range shall be copied to the output range.
 *
 *  This version of \p set_difference compares elements using \c operator<.
 *
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the output range.
 *  \return The end of the output range.
 *
 *  \tparam InputIterator1 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="http://www.sgi.com/tech/stl/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="http://www.sgi.com/tech/stl/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>.
 *
 *  The following code snippet demonstrates how to use \p set_difference to compute the
 *  set difference of two sets of integers sorted in ascending order.
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  ...
 *  int A1[6] = {0, 1, 3, 4, 5, 6, 9};
 *  int A2[5] = {1, 3, 5, 7, 9};
 *
 *  int result[3];
 *
 *  int *result_end = thrust::set_difference(A1, A1 + 6, A2, A2 + 5, result);
 *  // result is now {0, 4, 6}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/set_difference.html
 *  \see \p includes
 *  \see \p set_union
 *  \see \p set_intersection
 *  \see \p set_symmetric_difference
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator set_difference(InputIterator1 first1,
                                InputIterator1 last1,
                                InputIterator2 first2,
                                InputIterator2 last2,
                                OutputIterator result);

/*! \p set_difference constructs a sorted range that is the set difference of the sorted
 *  ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>. The return value is the
 *  end of the output range.
 *
 *  In the simplest case, \p set_difference performs the "difference" operation from set
 *  theory: the output range contains a copy of every element that is contained in
 *  <tt>[first1, last1)</tt> and not contained in <tt>[first2, last1)</tt>. The general case
 *  is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[first1, last1)</tt> contains \c m elements
 *  that are equivalent to each other and if <tt>[first2, last2)</tt> contains \c n
 *  elements that are equivalent to them, the last <tt>max(m-n,0)</tt> elements from
 *  <tt>[first1, last1)</tt> range shall be copied to the output range.
 *
 *  This version of \p set_difference compares elements using a function object \p comp.
 *
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the output range.
 *  \param comp Comparison operator.
 *  \return The end of the output range.
 *
 *  \tparam InputIterator1 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>,
 *          \p InputIterator1's \c value_type is convertable to \p StrictWeakCompare's \c first_argument_type.
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>,
 *          \p InputIterator2's \c value_type is convertable to \p StrictWeakCompare's \c second_argument_type.
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>.
 *  \tparam StrictWeakCompare is a model of <a href="http://www.sgi.com/tech/stl/StrictWeakOrdering.html">Strict Weak Ordering</a>.
 *
 *  The following code snippet demonstrates how to use \p set_difference to compute the
 *  set difference of two sets of integers sorted in descending order.
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  #include <thrust/functional.h>
 *  ...
 *  int A1[6] = {9, 6, 5, 4, 3, 1, 0};
 *  int A2[5] = {9, 7, 5, 3, 1};
 *
 *  int result[3];
 *
 *  int *result_end = thrust::set_difference(A1, A1 + 6, A2, A2 + 5, result, thrust::greater<int>());
 *  // result is now {6, 4, 0}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/set_difference.html
 *  \see \p includes
 *  \see \p set_union
 *  \see \p set_intersection
 *  \see \p set_symmetric_difference
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
  OutputIterator set_difference(InputIterator1 first1,
                                InputIterator1 last1,
                                InputIterator2 first2,
                                InputIterator2 last2,
                                OutputIterator result,
                                StrictWeakCompare comp);

/*! \p set_intersection constructs a sorted range that is the
 *  intersection of sorted ranges <tt>[first1, last1)</tt> and
 *  <tt>[first2, last2)</tt>. The return value is the end of the
 *  output range.
 *
 *  In the simplest case, \p set_intersection performs the
 *  "intersection" operation from set theory: the output range
 *  contains a copy of every element that is contained in both
 *  <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>. The
 *  general case is more complicated, because the input ranges may
 *  contain duplicate elements. The generalization is that if a value
 *  appears \c m times in <tt>[first1, last1)</tt> and \c n times in
 *  <tt>[first2, last2)</tt> (where \c m may be zero), then it
 *  appears <tt>min(m,n)</tt> times in the output range.
 *  \p set_intersection is stable, meaning that both elements are
 *  copied from the first range rather than the second, and that the
 *  relative order of elements in the output range is the same as in
 *  the first input range.
 *
 *  This version of \p set_intersection compares objects using
 *  \c operator<.
 *
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the output range.
 *  \return The end of the output range.
 *
 *  \tparam InputIterator1 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="http://www.sgi.com/tech/stl/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="http://www.sgi.com/tech/stl/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>.
 *
 *  The following code snippet demonstrates how to use \p set_intersection to compute the
 *  set intersection of two sets of integers sorted in ascending order.
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  ...
 *  int A1[6] = {1, 3, 5, 7, 9, 11};
 *  int A2[7] = {1, 1, 2, 3, 5,  8, 13};
 *
 *  int result[7];
 *
 *  int *result_end = thrust::set_intersection(A1, A1 + 6, A2, A2 + 7, result);
 *  // result is now {1, 3, 5}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/set_intersection.html
 *  \see \p includes
 *  \see \p set_union
 *  \see \p set_intersection
 *  \see \p set_symmetric_difference
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator set_intersection(InputIterator1 first1,
                                  InputIterator1 last1,
                                  InputIterator2 first2,
                                  InputIterator2 last2,
                                  OutputIterator result);

/*! \p set_intersection constructs a sorted range that is the
 *  intersection of sorted ranges <tt>[first1, last1)</tt> and
 *  <tt>[first2, last2)</tt>. The return value is the end of the
 *  output range.
 *
 *  In the simplest case, \p set_intersection performs the
 *  "intersection" operation from set theory: the output range
 *  contains a copy of every element that is contained in both
 *  <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>. The
 *  general case is more complicated, because the input ranges may
 *  contain duplicate elements. The generalization is that if a value
 *  appears \c m times in <tt>[first1, last1)</tt> and \c n times in
 *  <tt>[first2, last2)</tt> (where \c m may be zero), then it
 *  appears <tt>min(m,n)</tt> times in the output range.
 *  \p set_intersection is stable, meaning that both elements are
 *  copied from the first range rather than the second, and that the
 *  relative order of elements in the output range is the same as in
 *  the first input range.
 *
 *  This version of \p set_intersection compares elements using a function object \p comp.
 *
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the output range.
 *  \param comp Comparison operator.
 *  \return The end of the output range.
 *
 *  \tparam InputIterator1 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="http://www.sgi.com/tech/stl/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="http://www.sgi.com/tech/stl/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>.
 *
 *  The following code snippet demonstrates how to use \p set_intersection to compute
 *  the set intersection of sets of integers sorted in descending order.
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  ...
 *  int A1[6] = {11, 9, 7, 5, 3, 1};
 *  int A2[7] = {13, 8, 5, 3, 2,  1, 1};
 *
 *  int result[3];
 *
 *  int *result_end = thrust::set_intersection(A1, A1 + 6, A2, A2 + 7, result, thrust::greater<int>());
 *  // result is now {5, 3, 1}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/set_intersection.html
 *  \see \p includes
 *  \see \p set_union
 *  \see \p set_intersection
 *  \see \p set_symmetric_difference
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
  OutputIterator set_intersection(InputIterator1 first1,
                                  InputIterator1 last1,
                                  InputIterator2 first2,
                                  InputIterator2 last2,
                                  OutputIterator result,
                                  StrictWeakCompare comp);

/*! \p set_symmetric_difference constructs a sorted range that is the set symmetric
 *  difference of the sorted ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>.
 *  The return value is the end of the output range.
 *
 *  In the simplest case, \p set_symmetric_difference performs a set theoretic calculation:
 *  it constructs the union of the two sets A - B and B - A, where A and B are the two
 *  input ranges. That is, the output range contains a copy of every element that is
 *  contained in <tt>[first1, last1)</tt> but not <tt>[first2, last1)</tt>, and a copy of
 *  every element that is contained in <tt>[first2, last2)</tt> but not <tt>[first1, last1)</tt>.
 *  The general case is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[first1, last1)</tt> contains \c m elements that are
 *  equivalent to each other and <tt>[first2, last1)</tt> contains \c n elements that are
 *  equivalent to them, then <tt>|m - n|</tt> of those elements shall be copied to the output
 *  range: the last <tt>m - n</tt> elements from <tt>[first1, last1)</tt> if <tt>m > n</tt>, and
 *  the last <tt>n - m</tt> of these elements from <tt>[first2, last2)</tt> if <tt>m < n</tt>.
 *
 *  This version of \p set_union compares elements using \c operator<.
 *
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the output range.
 *  \return The end of the output range.
 *
 *  \tparam InputIterator1 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="http://www.sgi.com/tech/stl/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="http://www.sgi.com/tech/stl/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>.
 *
 *  The following code snippet demonstrates how to use \p set_symmetric_difference to compute
 *  the symmetric difference of two sets of integers sorted in ascending order.
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  ...
 *  int A1[6] = {0, 1, 2, 2, 4, 6, 7};
 *  int A2[5] = {1, 1, 2, 5, 8};
 *
 *  int result[6];
 *
 *  int *result_end = thrust::set_symmetric_difference(A1, A1 + 6, A2, A2 + 5, result);
 *  // result = {0, 4, 5, 6, 7, 8}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/set_symmetric_difference.html
 *  \see \p merge
 *  \see \p includes
 *  \see \p set_difference
 *  \see \p set_union
 *  \see \p set_intersection
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator set_symmetric_difference(InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          InputIterator2 last2,
                                          OutputIterator result);

/*! \p set_symmetric_difference constructs a sorted range that is the set symmetric
 *  difference of the sorted ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>.
 *  The return value is the end of the output range.
 *
 *  In the simplest case, \p set_symmetric_difference performs a set theoretic calculation:
 *  it constructs the union of the two sets A - B and B - A, where A and B are the two
 *  input ranges. That is, the output range contains a copy of every element that is
 *  contained in <tt>[first1, last1)</tt> but not <tt>[first2, last1)</tt>, and a copy of
 *  every element that is contained in <tt>[first2, last2)</tt> but not <tt>[first1, last1)</tt>.
 *  The general case is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[first1, last1)</tt> contains \c m elements that are
 *  equivalent to each other and <tt>[first2, last1)</tt> contains \c n elements that are
 *  equivalent to them, then <tt>|m - n|</tt> of those elements shall be copied to the output
 *  range: the last <tt>m - n</tt> elements from <tt>[first1, last1)</tt> if <tt>m > n</tt>, and
 *  the last <tt>n - m</tt> of these elements from <tt>[first2, last2)</tt> if <tt>m < n</tt>.
 *
 *  This version of \p set_union compares elements using a function object \p comp.
 *
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the output range.
 *  \param comp Comparison operator.
 *  \return The end of the output range.
 *
 *  \tparam InputIterator1 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="http://www.sgi.com/tech/stl/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="http://www.sgi.com/tech/stl/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>.
 *
 *  The following code snippet demonstrates how to use \p set_symmetric_difference to compute
 *  the symmetric difference of two sets of integers sorted in descending order.
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  ...
 *  int A1[6] = {7, 6, 4, 2, 2, 1, 0};
 *  int A2[5] = {8, 5, 2, 1, 1};
 *
 *  int result[6];
 *
 *  int *result_end = thrust::set_symmetric_difference(A1, A1 + 6, A2, A2 + 5, result);
 *  // result = {8, 7, 6, 5, 4, 0}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/set_symmetric_difference.html
 *  \see \p merge
 *  \see \p includes
 *  \see \p set_difference
 *  \see \p set_union
 *  \see \p set_intersection
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
  OutputIterator set_symmetric_difference(InputIterator1 first1,
                                          InputIterator1 last1,
                                          InputIterator2 first2,
                                          InputIterator2 last2,
                                          OutputIterator result,
                                          StrictWeakCompare comp);


/*! \p set_union constructs a sorted range that is the union of the sorted ranges
 *  <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>. The return value is the
 *  end of the output range.
 *
 *  In the simplest case, \p set_union performs the "union" operation from set
 *  theory: the output range contains a copy of every element that is contained in
 *  <tt>[first1, last1)</tt>, <tt>[first2, last1)</tt>, or both. The general case
 *  is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[first1, last1)</tt> contains \c m elements
 *  that are equivalent to each other and if <tt>[first2, last2)</tt> contains \c n
 *  elements that are equivalent to them, then all \c m elements from the first
 *  range shall be copied to the output range, in order, and then <tt>max(n - m, 0)</tt>
 *  elements from the second range shall be copied to the output, in order.
 *
 *  This version of \p set_union compares elements using \c operator<.
 *
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the output range.
 *  \return The end of the output range.
 *
 *  \tparam InputIterator1 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>,
 *          \p InputIterator1 and \p InputIterator2 have the same \c value_type,
 *          \p InputIterator1's \c value_type is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator1's \c value_type is a strict weak ordering, as defined in the <a href="http://www.sgi.com/tech/stl/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>,
 *          \p InputIterator2 and \p InputIterator1 have the same \c value_type,
 *          \p InputIterator2's \c value_type is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable">LessThan Comparable</a>,
 *          the ordering on \p InputIterator2's \c value_type is a strict weak ordering, as defined in the <a href="http://www.sgi.com/tech/stl/LessThanComparable">LessThan Comparable</a> requirements,
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>.
 *
 *  The following code snippet demonstrates how to use \p set_union to compute the union of
 *  two sets of integers sorted in ascending order.
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  ...
 *  int A1[6] = {0, 2, 4, 6, 8, 10, 12};
 *  int A2[5] = {1, 3, 5, 7, 9};
 *
 *  int result[11];
 *
 *  int *result_end = thrust::set_union(A1, A1 + 6, A2, A2 + 5, result);
 *  // result = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/set_union.html
 *  \see \p merge
 *  \see \p includes
 *  \see \p set_union
 *  \see \p set_intersection
 *  \see \p set_symmetric_difference
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator set_union(InputIterator1 first1,
                           InputIterator1 last1,
                           InputIterator2 first2,
                           InputIterator2 last2,
                           OutputIterator result);

/*! \p set_union constructs a sorted range that is the union of the sorted ranges
 *  <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>. The return value is the
 *  end of the output range.
 *
 *  In the simplest case, \p set_union performs the "union" operation from set
 *  theory: the output range contains a copy of every element that is contained in
 *  <tt>[first1, last1)</tt>, <tt>[first2, last1)</tt>, or both. The general case
 *  is more complicated, because the input ranges may contain duplicate elements.
 *  The generalization is that if <tt>[first1, last1)</tt> contains \c m elements
 *  that are equivalent to each other and if <tt>[first2, last2)</tt> contains \c n
 *  elements that are equivalent to them, then all \c m elements from the first
 *  range shall be copied to the output range, in order, and then <tt>max(n - m, 0)</tt>
 *  elements from the second range shall be copied to the output, in order.
 *
 *  This version of \p set_union compares elements using a function object \p comp.
 *
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the output range.
 *  \param comp Comparison operator.
 *  \return The end of the output range.
 *
 *  \tparam InputIterator1 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>,
 *          \p InputIterator1's \c value_type is convertable to \p StrictWeakCompare's \c first_argument_type.
 *          and \p InputIterator1's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>,
 *          \p InputIterator2's \c value_type is convertable to \p StrictWeakCompare's \c second_argument_type.
 *          and \p InputIterator2's \c value_type is convertable to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>.
 *  \tparam StrictWeakCompare is a model of <a href="http://www.sgi.com/tech/stl/StrictWeakOrdering.html">Strict Weak Ordering</a>.
 *
 *  The following code snippet demonstrates how to use \p set_union to compute the union of
 *  two sets of integers sorted in ascending order.
 *
 *  \code
 *  #include <thrust/set_operations.h>
 *  #include <thrust/functional.h>
 *  ...
 *  int A1[6] = {12, 10, 8, 6, 4, 2, 12};
 *  int A2[5] = {9, 7, 5, 3, 1};
 *
 *  int result[11];
 *
 *  int *result_end = thrust::set_union(A1, A1 + 6, A2, A2 + 5, result, thrust::greater<int>());
 *  // result = {12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/set_union.html
 *  \see \p merge
 *  \see \p includes
 *  \see \p set_union
 *  \see \p set_intersection
 *  \see \p set_symmetric_difference
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
  OutputIterator set_union(InputIterator1 first1,
                           InputIterator1 last1,
                           InputIterator2 first2,
                           InputIterator2 last2,
                           OutputIterator result,
                           StrictWeakCompare comp);

/*! \} // end set_operations
 */

} // end thrust

#include <thrust/detail/set_operations.inl>

