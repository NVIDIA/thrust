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

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/minimum_system.h>
#include <thrust/system/detail/generic/type_traits.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{

template<typename Tag>
__host__ __device__
  typename thrust::detail::disable_if<
    select_system1_exists<Tag>::value,
    Tag
  >::type
    select_system(Tag)
{
  return Tag();
} // end select_system()

template<typename Tag1, typename Tag2>
__host__ __device__
  typename thrust::detail::lazy_disable_if<
    select_system2_exists<Tag1,Tag2>::value,
    thrust::detail::minimum_system<Tag1,Tag2>
  >::type
    select_system(Tag1, Tag2)
{
  // for now, return minimum_system
  return typename thrust::detail::minimum_system<Tag1,Tag2>::type();
} // end select_system()

template<typename Tag1, typename Tag2, typename Tag3>
__host__ __device__
  typename thrust::detail::lazy_disable_if<
    select_system3_exists<Tag1,Tag2,Tag3>::value,
    thrust::detail::minimum_system<Tag1,Tag2,Tag3>
  >::type
    select_system(Tag1, Tag2, Tag3)
{
  // for now, return minimum_system
  return typename thrust::detail::minimum_system<Tag1,Tag2,Tag3>::type();
} // end select_system()

template<typename Tag1, typename Tag2, typename Tag3, typename Tag4>
__host__ __device__
  typename thrust::detail::lazy_disable_if<
    select_system4_exists<Tag1,Tag2,Tag3,Tag4>::value,
    thrust::detail::minimum_system<Tag1,Tag2,Tag3,Tag4>
  >::type
    select_system(Tag1, Tag2, Tag3, Tag4)
{
  // for now, return minimum_system
  return typename thrust::detail::minimum_system<Tag1,Tag2,Tag3,Tag4>::type();
} // end select_system()

// map a single any_system_tag to device_system_tag
inline __host__ __device__
thrust::device_system_tag select_system(thrust::any_system_tag)
{
  return thrust::device_system_tag();
} // end select_system()

} // end generic
} // end detail
} // end system
} // end thrust

