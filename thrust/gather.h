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


/*! \file gather.h
 *  \brief Defines the interface to a function which fills an array
 *         with an incoherent gather operation.
 */

#pragma once

#include <thrust/detail/config.h>

namespace thrust
{

/*! \addtogroup copying
 *  \{
 *  \addtogroup irregular_copying Irregular Copying
 *  \ingroup copying
 *  \{
 */

/*! \p gather copies elements from a source array into a destination range according 
 *  to a map. For each output iterator \c i in the range [\p first, \p last), the 
 *  value \p input[*(\p map + (\p i - \p first))] is assigned to \p *i. \p RandomAccessIterator
 *  must permit random access. Gather operations between host and device memory spaces are supported
 *  in both directions.
 *
 *  \param first Beginning of the destination range.
 *  \param last End of the destination range.
 *  \param map Beginning of the sequence of gather locations.
 *  \param input Beginning of the source range.
 *
 *  \tparam ForwardIterator must be a model of <a href="http://www.sgi.com/tech/stl/ForwardIterator.html">Forward Iterator</a>.
 *  \tparam InputIterator must be a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a> and \c InputIterator's \c value_type must be convertible to \c RandomAccessIterator's \c difference_type.
 *  \tparam RandomAccessIterator must be a model of <a href="http://www.sgi.com/tech/stl/RandomAccessIterator.html">Random Access iterator</a> and \c RandomAccessIterator's \c value_type must be convertible to \c ForwardIterator's \c value_type.
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
 *  thrust::gather(output.begin(), output.end(),
 *                  map, input);
 *  // output is now {0, 0, 0, 0, 0, 1, 1, 1, 1, 1}
 *  \endcode
 *
 *  \note \p gather is the inverse of thrust::scatter.
 */
template<typename ForwardIterator,
         typename InputIterator,
         typename RandomAccessIterator>
  void gather(ForwardIterator first,
              ForwardIterator last,
              InputIterator map,
              RandomAccessIterator input);

/*! \p gather_if conditionally copies elements from a source array into a destination 
 *  range according to a map. For each output iterator \c i in the range 
 *  [\p first, \p last) such that the value of *(\p stencil + (\c i - \p first)) is
 *  \c true, the value \p input[*(\p map + (\c i - \p first))] is assigned to *\c i.
 *  \p RandomAccessIterator must permit random access.  
 *
 *  \param first Beginning of the destination range.
 *  \param last End of the destination range.
 *  \param map Beginning of the sequence of gather locations.
 *  \param stencil Beginning of the sequence of predicate values.
 *  \param input Beginning of the source range.
 *
 *  \tparam ForwardIterator must be a model of <a href="http://www.sgi.com/tech/stl/ForwardIterator.html">Forward Iterator</a>.
 *  \tparam InputIterator1 must be a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a> and \c InputIterator1's \c value_type must be convertible to \c RandomAccessIterator's \c difference_type.
 *  \tparam InputIterator2 must be a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a> and \c InputIterator2's \c value_type must be convertible to \c bool.
 *  \tparam RandomAccessIterator must be a model of <a href="http://www.sgi.com/tech/stl/RandomAccessIterator.html">Random Access iterator</a> and \c RandomAccessIterator's \c value_type must be convertible to \c ForwardIterator's \c value_type.
 *
 *  \note \p gather_if is the inverse of thrust::scatter_if.
 */
template<typename ForwardIterator,
         typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
  void gather_if(ForwardIterator first,
                 ForwardIterator last,
                 InputIterator1 map,
                 InputIterator2 stencil,
                 RandomAccessIterator input);

/*! \p gather_if conditionally copies elements from a source array into a destination 
 *  range according to a map. For each output iterator \c i in the range 
 *  [\p first, \p last) such that the value of *(\p stencil + (\c i - \p first)) is \c true, 
 *  the value <tt>input[*(map + (i - first))]</tt> is assigned to *\p i. \p RandomAccessIterator
 *  must permit random access.  
 *
 *  \param first Beginning of the destination array.
 *  \param last End of the destination array.
 *  \param map Beginning of the sequence of gather locations.
 *  \param stencil Beginning of the sequence of predicate values.
 *  \param input Beginning of the source range.
 *  \param pred Predicate to apply to the stencil values.
 *
 *  \tparam ForwardIterator must be a model of <a href="http://www.sgi.com/tech/stl/ForwardIterator.html">Forward Iterator</a>.
 *  \tparam InputIterator1 must be a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a> and \c InputIterator1's \c value_type must be convertible to \c RandomAccessIterator's \c difference_type.
 *  \tparam InputIterator2 must be a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a> and \c InputIterator2's \c value_type must be convertible to \c Predicate's \c argument_type.
 *  \tparam RandomAccessIterator must be a model of <a href="http://www.sgi.com/tech/stl/RandomAccessIterator.html">Random Access iterator</a> and \c RandomAccessIterator's \c value_type must be convertible to \c ForwardIterator's \c value_type.
 *  \tparam Predicate must be a model of <a href="http://www.sgi.com/tech/stl/Predicate.html">Predicate</a>.
 *
 *  \note \p gather_if is the inverse of thrust::scatter_if.
 */
template<typename ForwardIterator,
         typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename Predicate>
  void gather_if(ForwardIterator first,
                 ForwardIterator last,
                 InputIterator1 map,
                 InputIterator2 stencil,
                 RandomAccessIterator input,
                 Predicate pred);

/*! \} // irregular_copying
 *  \} // copying
 */

}; // end namespace thrust

#include <thrust/detail/gather.inl>

