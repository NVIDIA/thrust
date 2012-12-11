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


/*! \file scatter.h
 *  \brief Irregular copying to a destination range
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/dispatchable.h>

namespace thrust
{


template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
  void scatter(const thrust::detail::dispatchable_base<System> &system,
               InputIterator1 first,
               InputIterator1 last,
               InputIterator2 map,
               RandomAccessIterator output);

template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator>
  void scatter_if(const thrust::detail::dispatchable_base<System> &system,
                  InputIterator1 first,
                  InputIterator1 last,
                  InputIterator2 map,
                  InputIterator3 stencil,
                  RandomAccessIterator output);

template<typename System,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator,
         typename Predicate>
  void scatter_if(const thrust::detail::dispatchable_base<System> &system,
                  InputIterator1 first,
                  InputIterator1 last,
                  InputIterator2 map,
                  InputIterator3 stencil,
                  RandomAccessIterator output,
                  Predicate pred);


/*! \addtogroup scattering
 *  \ingroup copying
 *  \{
 */

/*! \p scatter copies elements from a source range into an output array
 *  according to a map. For each iterator \c i in the range [\p first, \p last),
 *  the value \c *i is assigned to <tt>output[*(map + (i - first))]</tt>. The 
 *  output iterator must permit random access. If the same index 
 *  appears more than once in the range <tt>[map, map + (last - first))</tt>,
 *  the result is undefined.
 *
 *  \param first Beginning of the sequence of values to scatter.
 *  \param last End of the sequence of values to scatter.
 *  \param map  Beginning of the sequence of output indices.
 *  \param result Destination of the source elements.
 *
 *  \tparam InputIterator1 must be a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a> and \c InputIterator1's \c value_type must be convertible to \c RandomAccessIterator's \c value_type.
 *  \tparam InputIterator2 must be a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a> and \c InputIterator2's \c value_type must be convertible to \c RandomAccessIterator's \c difference_type.
 *  \tparam RandomAccessIterator must be a model of <a href="http://www.sgi.com/tech/stl/RandomAccessIterator.html">Random Access iterator</a>.
 *
 *  The following code snippet demonstrates how to use \p scatter to
 *  reorder a range.
 *
 *  \code
 *  #include <thrust/scatter.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  // mark even indices with a 1; odd indices with a 0
 *  int values[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
 *  thrust::device_vector<int> d_values(values, values + 10);
 *
 *  // scatter all even indices into the first half of the
 *  // range, and odd indices vice versa
 *  int map[10]   = {0, 5, 1, 6, 2, 7, 3, 8, 4, 9};
 *  thrust::device_vector<int> d_map(map, map + 10);
 *
 *  thrust::device_vector<int> d_output(10);
 *  thrust::scatter(d_values.begin(), d_values.end(),
 *                  d_map.begin(), d_output.begin());
 *  // d_output is now {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}
 *  \endcode
 *
 *  \note \p scatter is the inverse of thrust::gather.
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
  void scatter(InputIterator1 first,
               InputIterator1 last,
               InputIterator2 map,
               RandomAccessIterator result);


/*! \p scatter_if conditionally copies elements from a source range into an 
 *  output array according to a map. For each iterator \c i in the 
 *  range <tt>[first, last)</tt> such that <tt>*(stencil + (i - first))</tt> is
 *  true, the value \c *i is assigned to <tt>output[*(map + (i - first))]</tt>.
 *  The output iterator must permit random access. If the same index 
 *  appears more than once in the range <tt>[map, map + (last - first))</tt>
 *  the result is undefined.
 *
 *  \param first Beginning of the sequence of values to scatter.
 *  \param last End of the sequence of values to scatter.
 *  \param map Beginning of the sequence of output indices.
 *  \param stencil Beginning of the sequence of predicate values.
 *  \param output Beginning of the destination range.
 *
 *  \tparam InputIterator1 must be a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a> and \c InputIterator1's \c value_type must be convertible to \c RandomAccessIterator's \c value_type.
 *  \tparam InputIterator2 must be a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a> and \c InputIterator2's \c value_type must be convertible to \c RandomAccessIterator's \c difference_type.
 *  \tparam InputIterator3 must be a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a> and \c InputIterator3's \c value_type must be convertible to \c bool.
 *  \tparam RandomAccessIterator must be a model of <a href="http://www.sgi.com/tech/stl/RandomAccessIterator.html">Random Access iterator</a>.
 *
 *  \code
 *  #include <thrust/scatter.h>
 *  ...
 *  int V[8] = {10, 20, 30, 40, 50, 60, 70, 80};
 *  int M[8] = {0, 5, 1, 6, 2, 7, 3, 4};
 *  int S[8] = {1, 0, 1, 0, 1, 0, 1, 0};
 *  int D[8] = {0, 0, 0, 0, 0, 0, 0, 0};
 * 
 *  thrust::scatter_if(V, V + 8, M, S, D);
 * 
 *  // D contains [10, 30, 50, 70, 0, 0, 0, 0];
 *  \endcode
 *
 *  \note \p scatter_if is the inverse of thrust::gather_if.
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator>
  void scatter_if(InputIterator1 first,
                  InputIterator1 last,
                  InputIterator2 map,
                  InputIterator3 stencil,
                  RandomAccessIterator output);
                  

/*! \p scatter_if conditionally copies elements from a source range into an 
 *  output array according to a map. For each iterator \c i in the 
 *  range <tt>[first, last)</tt> such that <tt>pred(*(stencil + (i - first)))</tt> is
 *  \c true, the value \c *i is assigned to <tt>output[*(map + (i - first))]</tt>.
 *  The output iterator must permit random access. If the same index 
 *  appears more than once in the range <tt>[map, map + (last - first))</tt>
 *  the result is undefined.
 *
 *  \param first Beginning of the sequence of values to scatter.
 *  \param last End of the sequence of values to scatter.
 *  \param map Beginning of the sequence of output indices.
 *  \param stencil Beginning of the sequence of predicate values.
 *  \param output Beginning of the destination range.
 *  \param pred Predicate to apply to the stencil values.
 *
 *  \tparam InputIterator1 must be a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a> and \c InputIterator1's \c value_type must be convertible to \c RandomAccessIterator's \c value_type.
 *  \tparam InputIterator2 must be a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a> and \c InputIterator2's \c value_type must be convertible to \c RandomAccessIterator's \c difference_type.
 *  \tparam InputIterator3 must be a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a> and \c InputIterator3's \c value_type must be convertible to \c Predicate's \c argument_type.
 *  \tparam RandomAccessIterator must be a model of <a href="http://www.sgi.com/tech/stl/RandomAccessIterator.html">Random Access iterator</a>.
 *  \tparam Predicate must be a model of <a href="http://www.sgi.com/tech/stl/Predicate.html">Predicate</a>.
 *
 *  \code
 *  #include <thrust/scatter.h>
 *
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(int x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *
 *  ...
 *
 *  int V[8] = {10, 20, 30, 40, 50, 60, 70, 80};
 *  int M[8] = {0, 5, 1, 6, 2, 7, 3, 4};
 *  int S[8] = {2, 1, 2, 1, 2, 1, 2, 1};
 *  int D[8] = {0, 0, 0, 0, 0, 0, 0, 0};
 * 
 *  is_even pred;
 *  thrust::scatter_if(V, V + 8, M, S, D, pred);
 * 
 *  // D contains [10, 30, 50, 70, 0, 0, 0, 0];
 *  \endcode
 *  
 *  \note \p scatter_if is the inverse of thrust::gather_if.
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator,
         typename Predicate>
  void scatter_if(InputIterator1 first,
                  InputIterator1 last,
                  InputIterator2 map,
                  InputIterator3 stencil,
                  RandomAccessIterator output,
                  Predicate pred);

/*! \} // end scattering
 */

} // end namespace thrust

#include <thrust/detail/scatter.inl>

