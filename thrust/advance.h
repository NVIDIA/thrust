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


/*! \file advance.h
 *  \brief Advance an iterator by a given distance.
 */

#pragma once

#include <thrust/detail/config.h>

namespace thrust
{

/*! \addtogroup iterators
 *  \{
 */

/*! \p advance(i, n) increments the iterator \p i by the distance \p n. 
 *  If <tt>n > 0</tt> it is equivalent to executing <tt>++i</tt> \p n
 *  times, and if <tt>n < 0</tt> it is equivalent to executing <tt>--i</tt>
 *  \p n times. If <tt>n == 0</tt>, the call has no effect.
 *
 *  \param i The iterator to be advanced.
 *  \param n The distance by which to advance the iterator.
 *
 *  \tparam InputIterator is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>.
 *  \tparam Distance is an integral type that is convertible to \p InputIterator's distance type. 
 *
 *  The following code snippet demonstrates how to use \p advance to increment
 *  an iterator a given number of times.
 *
 *  \code
 *  #include <thrust/advance.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  thrust::device_vector<int> vec(13);
 *  thrust::device_vector<int>::iterator iter = vec.begin();
 *
 *  thrust::advance(iter, 7);
 *
 *  // iter - vec.begin() == 7
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/advance.html
 */
template <typename InputIterator, typename Distance>
void advance(InputIterator& i, Distance n);

/*! \} // end iterators
 */

} // end thrust

#include <thrust/detail/advance.inl>

