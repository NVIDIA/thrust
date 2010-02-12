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


/*! \file generate.h
 *  \brief Defines the interface to generate.
 */

#pragma once

#include <thrust/detail/config.h>

namespace thrust
{

/*! \addtogroup transformations
 *  \{
 */

/*! \p generate assigns the result of invoking \p gen, a function object that takes no arguments,
 *  to each element in the range <tt>[first,last)</tt>.
 *
 *  \param first The first element in the range of interest.
 *  \param last The last element in the range of interest.
 *  \param gen A function argument, taking no parameters, used to generate values to assign to
 *             elements in the range <tt>[first,last)</tt>.
 *
 *  \tparam ForwardIterator is a model of <a href="http://www.sgi.com/tech/stl/ForwardIterator.html">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable.
 *  \tparam Generator is a model of <a href="http://www.sgi.com/tech/stl/Generator.html">Generator</a>,
 *          and \p Generator's \c result_type is convertible to \p ForwardIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to fill a \c host_vector with random numbers,
 *  using the standard C library function \c rand.
 *
 *  \code
 *  #include <thrust/generate.h>
 *  #include <thrust/host_vector.h>
 *  #include <stdlib.h>
 *  ...
 *  thrust::host_vector<int> v(10);
 *  srand(13);
 *  thrust::generate(v.begin(), v.end(), rand);
 *
 *  // the elements of v are now with pseudo-random numbers
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/generate.html
 */
template<typename ForwardIterator,
         typename Generator>
  void generate(ForwardIterator first,
                ForwardIterator last,
                Generator gen);

/*! \} // end transformations
 */

} // end namespace thrust

#include <thrust/detail/generate.inl>

