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

#include <thrust/detail/config.h>
#include <thrust/system/tbb/detail/default_decomposition.h>
#include <tbb/compat/thread>

namespace thrust
{
namespace system
{
namespace tbb
{
namespace detail
{

template <typename IndexType>
thrust::system::detail::internal::uniform_decomposition<IndexType> default_decomposition(IndexType n)
{
  return thrust::system::detail::internal::uniform_decomposition<IndexType>(n, 1, std::thread::hardware_concurrency());
}

template <typename IndexType>
thrust::system::detail::internal::uniform_decomposition<IndexType> default_decomposition(tag, IndexType n)
{
  return default_decomposition(n);
}

} // end namespace detail
} // end namespace tbb
} // end namespace system
} // end namespace thrust

