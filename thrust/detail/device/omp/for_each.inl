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


/*! \file for_each.inl
 *  \brief Inline file for for_each.h.
 */

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
  typedef typename thrust::iterator_difference<InputIterator>::type difference;
  difference n = thrust::distance(first,last);

#pragma omp parallel for
  for(difference i = 0;
      i < n;
      ++i)
  {
    f(thrust::detail::device::dereference(first, i));
  }
} 


} // end namespace omp

} // end namespace device

} // end namespace detail

} // end namespace thrust

