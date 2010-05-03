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


/*! \file gather.h
 *  \brief Defines the interface to a function which fills an array
 *         with an incoherent gather operation.
 */

#pragma once

#include <thrust/detail/config.h>

namespace thrust
{

/*! \addtogroup gathering
 *  \ingroup copying
 *  \{
 */

/*! \p gather copies elements from a source array into a destination range according 
 *  to a map. For each input iterator \c i in the range <tt>[map_first, map_last)</tt>, the
 *  value <tt>input_first[*i]</tt> is assigned to <tt>*(result + (i - map_first))</tt>.
 *  \p RandomAccessIterator must permit random access. Gather operations between host and device memory
 *  spaces are supported in both directions.
 *
 *  \param map_first Beginning of the range of gather locations.
 *  \param map_last End of the range of gather locations.
 *  \param input_first Beginning of the source range.
 *  \param result Beginning of the destination range.
 *
 *  \tparam InputIterator must be a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a> and \c InputIterator's \c value_type must be convertible to \c RandomAccessIterator's \c difference_type.
 *  \tparam RandomAccessIterator must be a model of <a href="http://www.sgi.com/tech/stl/RandomAccessIterator.html">Random Access Iterator</a> and \c RandomAccessIterator's \c value_type must be convertible to \c OutputIterator's \c value_type.
 *  \tparam OutputIterator must be a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>.
 *
 *  The following code snippet demonstrates how to use \p gather to reorder
 *  a range.
 *
 *  \code
 *  #include <thrust/gather.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  // mark odd indices with a 1; even indices with a 0
 *  int input[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
 *
 *  // gather all odd indices into the first half of the
 *  // range, and even indices vice versa
 *  int map[10]   = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};
 *
 *  thrust::device_vector<int> output(10);
 *  thrust::gather(map, map + 10, input, output.begin());
 *  // output is now {0, 0, 0, 0, 0, 1, 1, 1, 1, 1}
 *  \endcode
 *
 *  \note \p gather is the inverse of thrust::scatter.
 *
 *  \deprecated This function is is scheduled for promotion to \p thrust::gather in Thrust v1.3.
 */
template<typename InputIterator,
         typename RandomAccessIterator,
         typename OutputIterator>
  OutputIterator gather(InputIterator        map_first,
                        InputIterator        map_last,
                        RandomAccessIterator input_first,
                        OutputIterator       result);


/*! \p gather_if conditionally copies elements from a source array into a destination 
 *  range according to a map. For each input iterator \c i in the range <tt>[map_first, map_last)</tt>,
 *  such that the value of <tt>*(stencil + (i - map_first))</tt> is \c true, the value
 *  <tt>input_first[*i]</tt> is assigned to <tt>*(result + (i - map_first))</tt>.
 *  \p RandomAccessIterator must permit random access.
 *
 *  \param map_first Beginning of the range of gather locations.
 *  \param map_last End of the range of gather locations.
 *  \param stencil Beginning of the range of predicate values.
 *  \param input_first Beginning of the source range.
 *  \param result Beginning of the destination range.
 *
 *  \tparam InputIterator1 must be a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a> and \c InputIterator1's \c value_type must be convertible to \c RandomAccessIterator's \c difference_type.
 *  \tparam InputIterator2 must be a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a> and \c InputIterator2's \c value_type must be convertible to \c bool.
 *  \tparam RandomAccessIterator must be a model of <a href="http://www.sgi.com/tech/stl/RandomAccessIterator.html">Random Access iterator</a> and \c RandomAccessIterator's \c value_type must be convertible to \c OutputIterator's \c value_type.
 *  \tparam OutputIterator must be a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>.
 *
 *  \note \p gather_if is the inverse of thrust::scatter_if.
 *
 *  \deprecated This function is is scheduled for promotion to \p thrust::gather_if in Thrust v1.3.
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename OutputIterator>
  OutputIterator gather_if(InputIterator1       map_first,
                           InputIterator1       map_last,
                           InputIterator2       stencil,
                           RandomAccessIterator input_first,
                           OutputIterator       result);


/*! \p gather_if conditionally copies elements from a source array into a destination 
 *  range according to a map. For each input iterator \c i in the range <tt>[map_first, map_last)</tt>
 *  such that the value of <tt>pred(*(stencil + (i - map_first)))</tt> is \c true,
 *  the value <tt>input_first[*i]</tt> is assigned to <tt>*(result + (i - map_first))</tt>.
 *  \p RandomAccessIterator must permit random access.
 *
 *  \param map_first Beginning of the range of gather locations.
 *  \param map_last End of the range of gather locations.
 *  \param stencil Beginning of the range of predicate values.
 *  \param input_first Beginning of the source range.
 *  \param result Beginning of the destination range.
 *  \param pred Predicate to apply to the stencil values.
 *
 *  \tparam InputIterator1 must be a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a> and \c InputIterator1's \c value_type must be convertible to \c RandomAccessIterator's \c difference_type.
 *  \tparam InputIterator2 must be a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a> and \c InputIterator2's \c value_type must be convertible to \c Predicate's \c argument_type.
 *  \tparam RandomAccessIterator must be a model of <a href="http://www.sgi.com/tech/stl/RandomAccessIterator.html">Random Access iterator</a> and \c RandomAccessIterator's \c value_type must be convertible to \c OutputIterator's \c value_type.
 *  \tparam OutputIterator must be a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>.
 *  \tparam Predicate must be a model of <a href="http://www.sgi.com/tech/stl/Predicate.html">Predicate</a>.
 *
 *  \note \p gather_if is the inverse of thrust::scatter_if.
 *
 *  \deprecated This function is is scheduled for promotion to \p thrust::gather_if in Thrust v1.3.
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator gather_if(InputIterator1       map_first,
                           InputIterator1       map_last,
                           InputIterator2       stencil,
                           RandomAccessIterator input_first,
                           OutputIterator       result,
                           Predicate            pred);

/*! \} // gathering
 */

}; // end namespace thrust

#include <thrust/detail/gather.inl>

