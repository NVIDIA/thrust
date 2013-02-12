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
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/system/cuda/detail/assign_value.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{


namespace
{


template<typename DerivedPolicy, typename Pointer>
inline __host__ __device__
  typename thrust::iterator_value<Pointer>::type
    get_value_msvc2005_war(execution_policy<DerivedPolicy> &exec, Pointer ptr)
{
  typedef typename thrust::iterator_value<Pointer>::type result_type;

  // XXX war nvbugs/881631
  struct war_nvbugs_881631
  {
    __host__ inline static result_type host_path(execution_policy<DerivedPolicy> &exec, Pointer ptr)
    {
      // when called from host code, implement with assign_value
      // note that this requires a type with default constructor
      result_type result;

      thrust::host_system_tag host_tag;
      cross_system<thrust::host_system_tag, DerivedPolicy> systems(host_tag, exec);
      assign_value(systems, &result, ptr);

      return result;
    }

    __device__ inline static result_type device_path(execution_policy<DerivedPolicy> &, Pointer ptr)
    {
      // when called from device code, just do simple deref
      return *thrust::raw_pointer_cast(ptr);
    }
  };

#ifndef __CUDA_ARCH__
  return war_nvbugs_881631::host_path(exec, ptr);
#else
  return war_nvbugs_881631::device_path(exec, ptr);
#endif // __CUDA_ARCH__
} // end get_value_msvc2005_war()


} // end anon namespace


template<typename DerivedPolicy, typename Pointer>
inline __host__ __device__
  typename thrust::iterator_value<Pointer>::type
    get_value(execution_policy<DerivedPolicy> &exec, Pointer ptr)
{
  return get_value_msvc2005_war(exec,ptr);
} // end get_value()


} // end detail
} // end cuda
} // end system
} // end thrust

