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
#include <thrust/system/detail/internal/default_decomposition.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/internal/default_decomposition_adl_helper.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace internal
{

template <typename System, typename IndexType>
uniform_decomposition<IndexType> default_decomposition(IndexType n)
{
  return default_decomposition(System(), n);
} // end default_decomposition()

} // end namespace internal
} // end namespace detail
} // end namespace system
} // end namespace thrust

