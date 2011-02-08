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


/*! \file uninitialized_copy.h
 *  \brief Defines the interface to the
 *         uninitialized_copy function.
 *  \see http://www.sgi.com/tech/stl/uninitialized_copy.html
 */

#pragma once

#include <thrust/detail/config.h>

namespace thrust
{

/*! \addtogroup copying
 *  \{
 */

/*! In \c thrust, the function \c thrust::device_new allocates memory for
 *  an object and then creates an object at that location by calling a constructor.
 *  Occasionally, however, it is useful to separate those two operations.
 *  If each iterator in the range <tt>[result, result + (last - first))</tt> points
 *  to uninitialized memory, then \p uninitialized_copy creates a copy of
 *  <tt>[first, last)</tt> in that range. That is, for each iterator \c i in
 *  the input, \p uninitialized_copy creates a copy of \c *i in the location pointed
 *  to by the corresponding iterator in the output range by \p ForwardIterator's
 *  \c value_type's copy constructor with *i as its argument.
 *
 *  \param first The first element of the input range to copy from.
 *  \param last The last element of the input range to copy from.
 *  \param result The first element of the output range to copy to.
 *  \return An iterator pointing to the last element of the output range.
 *
 *  \tparam InputIterator is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>.
 *  \tparam ForwardIterator is a model of <a href="http://www.sgi.com/tech/stl/ForwardIterator.html">Forward Iterator</a>,
 *          \p ForwardIterator is mutable, and \p ForwardIterator's \c value_type has a constructor that takes
 *          a single argument whose type is \p InputIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p uninitialized_copy to initialize
 *  a range of uninitialized memory.
 *
 *  \code
 *  #include <thrust/uninitialized_copy.h>
 *  #include <thrust/device_malloc.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  struct Int
 *  {
 *    __host__ __device__
 *    Int(int x) : val(x) {}
 *    int val;
 *  };  
 *  ...
 *  const int N = 137;
 *
 *  Int val(46);
 *  thrust::device_vector<Int> input(N, val);
 *  thrust::device_ptr<Int> array = thrust::device_malloc<Int>(N);
 *  thrust::uninitialized_copy(input.begin(), input.end(), array);
 *
 *  // Int x = array[i];
 *  // x.val == 46 for all 0 <= i < N
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/uninitialized_copy.html
 *  \see \c copy
 *  \see \c uninitialized_fill
 *  \see \c device_new
 *  \see \c device_malloc
 */
template<typename InputIterator, typename ForwardIterator>
  ForwardIterator uninitialized_copy(InputIterator first,
                                     InputIterator last,
                                     ForwardIterator result);

/*! \} // copying
 */

} // end thrust

#include <thrust/detail/uninitialized_copy.inl>

