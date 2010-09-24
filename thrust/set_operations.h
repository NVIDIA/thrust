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
 *  The following code snippet demonstrates how to use
 *  \p set_intersection to compute the intersection of two sorted
 *  sets of integers.
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
 *  // result[0] = 1
 *  // result[1] = 3
 *  // result[2] = 5
 *  // values beyond result[2] are undefined
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/set_intersection.html
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

/*! \} // end set_operations
 */

} // end thrust

#include <thrust/detail/set_operations.inl>

