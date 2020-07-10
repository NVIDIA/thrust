/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

#if THRUST_CPP_DIALECT >= 2011
  #include <thrust/detail/execute_with_dependencies.h>
#endif
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  #include <thrust/detail/execute_with_stream.h>
#endif

namespace thrust
{
namespace detail
{

template <typename Allocator, template <typename> class BaseSystem>
struct execute_with_allocator
  : BaseSystem<execute_with_allocator<Allocator, BaseSystem> >
{
private:
  typedef BaseSystem<execute_with_allocator<Allocator, BaseSystem> > super_t;

  Allocator alloc;

public:
  __host__ __device__
  execute_with_allocator(super_t const& super, Allocator alloc_)
    : super_t(super), alloc(alloc_)
  {}

  __thrust_exec_check_disable__
  __host__ __device__
  execute_with_allocator(Allocator alloc_)
    : alloc(alloc_)
  {}

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  __host__
  execute_with_allocator_and_stream<Allocator, BaseSystem>
  on(cudaStream_t stream) const
  {
    return { alloc, stream };
  }
#endif

#if THRUST_CPP_DIALECT >= 2011
  template<typename ...Dependencies>
  __host__
  execute_with_allocator_and_dependencies<Allocator, BaseSystem, Dependencies...>
  after(Dependencies&& ...dependencies) const
  {
    return { alloc, capture_as_dependency(THRUST_FWD(dependencies))... };
  }

  template<typename ...Dependencies>
  __host__
  execute_with_allocator_and_dependencies<Allocator, BaseSystem, Dependencies...>
  after(std::tuple<Dependencies...>& dependencies) const
  {
    return { alloc, capture_as_dependency(dependencies) };
  }
  template<typename ...Dependencies>
  __host__
  execute_with_allocator_and_dependencies<Allocator, BaseSystem, Dependencies...>
  after(std::tuple<Dependencies...>&& dependencies) const
  {
    return { alloc, capture_as_dependency(std::move(dependencies)) };
  }

  template<typename ...Dependencies>
  __host__
  execute_with_allocator_and_dependencies<Allocator, BaseSystem, Dependencies...>
  rebind_after(Dependencies&& ...dependencies) const
  {
    return after(THRUST_FWD(dependencies)...);
  }

  template<typename ...Dependencies>
  __host__
  execute_with_allocator_and_dependencies<Allocator, BaseSystem, Dependencies...>
  rebind_after(std::tuple<Dependencies...>& dependencies) const
  {
    return after(dependencies);
  }
  template<typename ...Dependencies>
  __host__
  execute_with_allocator_and_dependencies<Allocator, BaseSystem, Dependencies...>
  rebind_after(std::tuple<Dependencies...>&& dependencies) const
  {
    return after(std::move(dependencies));
  }
#endif

  friend __host__
  typename remove_reference<Allocator>::type&
  dispatch_get_allocator(execute_with_allocator const& system)
  {
    return system.alloc;
  }
};

template <typename Derived>
__host__ auto
get_allocator(thrust::detail::execution_policy_base<Derived> const& policy)
THRUST_DECLTYPE_RETURNS(dispatch_get_allocator(derived_cast(policy)));

}} // namespace thrust::detail
