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


/*! \file for_each.inl
 *  \brief Inline file for for_each.h.
 */

#include <thrust/detail/config.h>
#include <thrust/detail/static_assert.h>
#include <thrust/detail/device/dereference.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/distance.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace omp
{

template<typename InputIterator,
         typename UnaryFunction>
void for_each(InputIterator first,
              InputIterator last,
              UnaryFunction f)
{
  // we're attempting to launch an omp kernel, assert we're compiling with omp support
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to OpenMP support in your compiler.                         X
  // ========================================================================
  THRUST_STATIC_ASSERT( (depend_on_instantiation<InputIterator,
                        (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)>::value) );

  typedef typename thrust::iterator_difference<InputIterator>::type difference;
  difference n = thrust::distance(first,last);

// do not attempt to compile the body of this function, which depends on #pragma omp,
// without support from the compiler
// XXX implement the body of this function in another file to eliminate this ugliness
#if (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
#pragma omp parallel for
  for(difference i = 0;
      i < n;
      ++i)
  {
    InputIterator temp = first + i;
    f(dereference(temp));
  }
#endif // THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE
} 


} // end namespace omp
} // end namespace device
} // end namespace detail
} // end namespace thrust

