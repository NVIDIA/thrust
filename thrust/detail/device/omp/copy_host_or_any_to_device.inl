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

#pragma once

// do not attempt to compile this code unless the compiler is generating multicore code
// TODO: do this check inside omp::copy_host_or_any_to_device with static_assert
// TODO: eliminate the need for this function once we have done away with device::dereference()
#ifdef _OPENMP

#include <thrust/distance.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/device/dereference.h>

namespace thrust
{

namespace detail
{

namespace device
{

namespace omp
{

template<typename InputIterator,
         typename OutputIterator>
OutputIterator copy_host_or_any_to_device(InputIterator first,
                                          InputIterator last,
                                          OutputIterator result)
{
  typedef typename thrust::iterator_difference<InputIterator>::type difference;
  difference n = thrust::distance(first,last);

#pragma omp parallel for
  for(difference i = 0;
      i < n;
      ++i)
  {
    OutputIterator temp = result + i;
    dereference(temp) = first[i];
  }

  return result + n;
}

} // end omp

} // end device

} // end detail

} // end thrust

#endif // _OPENMP


