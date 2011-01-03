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


#include <thrust/detail/device/dispatch/fill.h>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/distance.h>

namespace thrust
{
namespace detail
{
namespace device
{

template<typename ForwardIterator, typename T>
  void fill(ForwardIterator first,
            ForwardIterator last,
            const T &value)
{
  // this is safe because all device iterators are
  // random access at the moment
  thrust::detail::device::fill_n(first, thrust::distance(first,last), value);
}

template<typename OutputIterator, typename Size, typename T>
  OutputIterator fill_n(OutputIterator first,
                        Size n,
                        const T &value)
{
  // dispatch on space
  return thrust::detail::device::dispatch::fill_n(first, n, value,
    typename thrust::iterator_space<OutputIterator>::type());
}

} // end namespace device
} // end namespace detail
} // end namespace thrust

