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

/*! \file merge.h
 *  \brief Merging sorted ranges
 */

#pragma once

#include <thrust/detail/config.h>

namespace thrust
{

/*! \addtogroup merging Merging
 *  \ingroup algorithms
 *  \{
 */

/*! \p merge combines two sorted ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>
 *  into a single sorted range. That is, it copies from <tt>[first1, last1)</tt> and
 *  <tt>[first2, last2)</tt> into <tt>[result, result + (last1 - first1) + (last2 - first2))</tt>
 *  such that the resulting range is in ascending order. \p merge is stable, meaning both that the
 *  relative order of elements within each input range is preserved, and that for equivalent elements
 *  in both input ranges the element from the first range precedes the element from the second. The
 *  return value is <tt>result + (last1 - first1) + (last2 - first2)</tt>.
 *
 *  This version of \p merge compares elements using \c operator<.
 *
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the merged output.
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
 *  The following code snippet demonstrates how to use
 *  \p merge to compute the merger of two sorted sets of integers.
 *
 *  \code
 *  #include <thrust/merge.h>
 *  ...
 *  int A1[6] = {1, 3, 5, 7, 9, 11};
 *  int A2[7] = {1, 1, 2, 3, 5,  8, 13};
 *
 *  int result[13];
 *
 *  int *result_end = thrust::merge(A1, A1 + 6, A2, A2 + 7, result);
 *  // result = {1, 1, 1, 2, 3, 3, 5, 5, 7, 8, 9, 11, 13}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/merge.html
 *  \see \p set_union
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator merge(InputIterator1 first1,
                       InputIterator1 last1,
                       InputIterator2 first2,
                       InputIterator2 last2,
                       OutputIterator result);

/*! \p merge combines two sorted ranges <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>
 *  into a single sorted range. That is, it copies from <tt>[first1, last1)</tt> and
 *  <tt>[first2, last2)</tt> into <tt>[result, result + (last1 - first1) + (last2 - first2))</tt>
 *  such that the resulting range is in ascending order. \p merge is stable, meaning both that the
 *  relative order of elements within each input range is preserved, and that for equivalent elements
 *  in both input ranges the element from the first range precedes the element from the second. The
 *  return value is <tt>result + (last1 - first1) + (last2 - first2)</tt>.
 *
 *  This version of \p merge compares elements using a function object \p comp.
 *
 *  \param first1 The beginning of the first input range.
 *  \param last1 The end of the first input range.
 *  \param first2 The beginning of the second input range.
 *  \param last2 The end of the second input range.
 *  \param result The beginning of the merged output.
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
 *  The following code snippet demonstrates how to use
 *  \p merge to compute the merger of two sets of integers sorted in
 *  ascending order.
 *
 *  \code
 *  #include <thrust/merge.h>
 *  #include <thrust/functional.h>
 *  ...
 *  int A1[6] = {11, 9, 7, 5, 3, 1};
 *  int A2[7] = {13, 8, 5, 3, 2, 1, 1};
 *
 *  int result[13];
 *
 *  int *result_end = thrust::merge(A1, A1 + 6, A2, A2 + 7, result, thrust::greater<int>());
 *  // result = {13, 11, 9, 8, 7, 5, 5, 3, 3, 2, 1, 1, 1}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/merge.html
 *  \see \p sort
 *  \see \p is_sorted
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakCompare>
  OutputIterator merge(InputIterator1 first1,
                       InputIterator1 last1,
                       InputIterator2 first2,
                       InputIterator2 last2,
                       OutputIterator result,
                       StrictWeakCompare comp);

/*! \} // merging
 */

} // end thrust

#include <thrust/detail/merge.inl>

