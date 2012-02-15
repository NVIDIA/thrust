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


/*! \file gather.h
 *  \brief Irregular copying from a source range
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
 *  \p RandomAccessIterator must permit random access.
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
 *  // mark even indices with a 1; odd indices with a 0
 *  int values[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
 *  thrust::device_vector<int> d_values(values, values + 10);
 *
 *  // gather all even indices into the first half of the range
 *  // and odd indices to the last half of the range
 *  int map[10]   = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};
 *  thrust::device_vector<int> d_map(map, map + 10);
 *
 *  thrust::device_vector<int> d_output(10);
 *  thrust::gather(d_map.begin(), d_map.end(),
 *                 d_values.begin(),
 *                 d_output.begin());
 *  // d_output is now {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}
 *  \endcode
 *
 *  \note \p gather is the inverse of thrust::scatter.
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
 *  The following code snippet demonstrates how to use \p gather_if to gather selected values from
 *  an input range.
 *
 *  \code
 *  #include <thrust/gather.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *
 *  int values[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
 *  thrust::device_vector<int> d_values(values, values + 10);
 *
 *  // select elements at even-indexed locations
 *  int stencil[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
 *  thrust::device_vector<int> d_stencil(stencil, stencil + 10);
 *
 *  // map all even indices into the first half of the range
 *  // and odd indices to the last half of the range
 *  int map[10]   = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};
 *  thrust::device_vector<int> d_map(map, map + 10);
 *
 *  thrust::device_vector<int> d_output(10, 7);
 *  thrust::gather_if(d_map.begin(), d_map.end(),
 *                    d_stencil.begin(),
 *                    d_values.begin(),
 *                    d_output.begin());
 *  // d_output is now {0, 2, 4, 6, 8, 7, 7, 7, 7, 7}
 *  \endcode
 *
 *  \note \p gather_if is the inverse of \p scatter_if.
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
 *  The following code snippet demonstrates how to use \p gather_if to gather selected values from
 *  an input range based on an arbitrary selection function.
 *
 *  \code
 *  #include <thrust/gather.h>
 *  #include <thrust/device_vector.h>
 *  
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(const int x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *  ...
 *
 *  int values[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
 *  thrust::device_vector<int> d_values(values, values + 10);
 *
 *  // we will select an element when our stencil is even
 *  int stencil[10] = {0, 3, 4, 1, 4, 1, 2, 7, 8, 9};
 *  thrust::device_vector<int> d_stencil(stencil, stencil + 10);
 *
 *  // map all even indices into the first half of the range
 *  // and odd indices to the last half of the range
 *  int map[10]   = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};
 *  thrust::device_vector<int> d_map(map, map + 10);
 *
 *  thrust::device_vector<int> d_output(10, 7);
 *  thrust::gather_if(d_map.begin(), d_map.end(),
 *                    d_stencil.begin(),
 *                    d_values.begin(),
 *                    d_output.begin(),
 *                    is_even());
 *  // d_output is now {0, 7, 4, 7, 8, 7, 3, 7, 7, 7}
 *  \endcode
 *
 *  \note \p gather_if is the inverse of \p scatter_if.
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

