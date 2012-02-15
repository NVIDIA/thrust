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


#include <thrust/detail/backend/fill.h>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/backend/cpp/fill.h>
#include <thrust/detail/backend/cuda/fill.h>
#include <thrust/detail/backend/generic/fill.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace dispatch
{



template<typename ForwardIterator, typename T>
  void fill(ForwardIterator first,
            ForwardIterator last,
            const T &value,
            thrust::host_space_tag)
{
  thrust::detail::backend::cpp::fill(first, last, value);
}

template<typename ForwardIterator, typename T, typename Backend>
  void fill(ForwardIterator first,
            ForwardIterator last,
            const T &value,
            Backend)
{
  thrust::detail::backend::generic::fill(first, last, value);
}


template<typename OutputIterator, typename Size, typename T, typename Backend>
  OutputIterator fill_n(OutputIterator first,
                        Size n,
                        const T &value,
                        Backend)
{
  return thrust::detail::backend::generic::fill_n(first, n, value);
}

template<typename OutputIterator, typename Size, typename T>
  OutputIterator fill_n(OutputIterator first,
                        Size n,
                        const T &value,
                        thrust::detail::cuda_device_space_tag)
{
  return thrust::detail::backend::cuda::fill_n(first, n, value);
}



} // end namespace dispatch


template<typename ForwardIterator, typename T>
  void fill(ForwardIterator first,
            ForwardIterator last,
            const T &value)
{
  thrust::detail::backend::dispatch::fill(first, last, value,
    typename thrust::iterator_space<ForwardIterator>::type());
}

template<typename OutputIterator, typename Size, typename T>
  OutputIterator fill_n(OutputIterator first,
                        Size n,
                        const T &value)
{
  return thrust::detail::backend::dispatch::fill_n(first, n, value,
    typename thrust::iterator_space<OutputIterator>::type());
}

} // end namespace backend
} // end namespace detail
} // end namespace thrust

