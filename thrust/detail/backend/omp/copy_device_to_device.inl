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

#pragma once

// TODO: eliminate the need for this function once we have done away with device::dereference()

#include <thrust/detail/config.h>
#include <thrust/distance.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/backend/dereference.h>

namespace thrust
{

namespace detail
{

namespace backend
{

namespace omp
{

template<typename InputIterator,
         typename OutputIterator>
OutputIterator copy_device_to_device(InputIterator first,
                                     InputIterator last,
                                     OutputIterator result)
{
  typedef typename thrust::iterator_difference<InputIterator>::type difference;
  // difference n = thrust::distance(first,last); // XXX WAR crash VS2008 (64-bit)
  difference n = last - first;

// are we compiling with omp support?
// if no, we serialize on the "host"
#if (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
#pragma omp parallel for
#endif // omp support
  for(difference i = 0;
      i < n;
      ++i)
  {
    InputIterator  first_temp  = first  + i;
    OutputIterator result_temp = result + i;

    dereference(result_temp) = dereference(first_temp);
  }

  return result + n;
}

} // end omp

} // end backend

} // end detail

} // end thrust

