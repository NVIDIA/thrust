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


/*! \file utility.h
 *  \brief Defines utility functions
 *         too minor for own their own header.
 */

#pragma once

#include <thrust/detail/config.h>

/*!
 * dummy comment here so namespace thrust's documentation will be extracted
 */
namespace thrust
{

/*! \addtogroup utility
 *  \{
 */

/*! \p swap assigns the contents of \c a to \c b and the
 *  contents of \c b to \c a. This is used as a primitive operation
 *  by many other algorithms.
 *  
 *  \param a The first value of interest. After completion,
 *           the value of b will be returned here.
 *  \param b The second value of interest. After completion,
 *           the value of a will be returned here.
 *
 *  \tparam Assignable is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>.
 *
 *  The following code snippet demonstrates how to use \p swap to
 *  swap the contents of two variables.
 *
 *  \code
 *  #include <thrust/utility.h>
 *  ...
 *  int x = 1;
 *  int y = 2;
 *  thrust::swap(x,h);
 *
 *  // x == 2, y == 1
 *  \endcode
 */
template<typename Assignable1, typename Assignable2>
__host__ __device__ 
inline void swap(Assignable1 &a, Assignable2 &b);

/*! \} // utility
 */

} // end thrust

#include <thrust/detail/utility.inl>

