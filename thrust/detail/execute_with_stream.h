/*
 *  Copyright 2018 NVIDIA Corporation
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
#include <thrust/detail/execute_with_dependencies.h>
#include <thrust/system/cuda/stream.h> // FIXME: This is heavyweight.

namespace thrust
{
namespace detail
{

template<template<typename> class BaseSystem>
struct execute_with_stream : BaseSystem<execute_with_stream<BaseSystem>>
{
private:
  using super_t = BaseSystem<execute_with_stream<BaseSystem>>;

  cudaStream_t const stream;

public:
  // This must be __host__ __device__, because `.on` has to work in device
  // code, although `.after` explicitly does not work in device code.
  __host__ __device__
  execute_with_stream(super_t const &super, cudaStream_t stream)
    : super_t(super), stream(stream)
  {}

  // This must be __host__ __device__, because `.on` has to work in device
  // code, although `.after` explicitly does not work in device code.
  __thrust_exec_check_disable__
  __host__ __device__
  execute_with_stream(cudaStream_t stream)
    : stream(stream)
  {}

  template<typename ...Dependencies>
  __host__
  execute_with_dependencies<
    BaseSystem, cuda::unique_stream, Dependencies...
  >
  after(Dependencies&& ...dependencies) const
  {
    return { std::make_tuple(
      cuda::unique_stream(cuda::nonowning, stream),
      capture_as_dependency(THRUST_FWD(dependencies))...
    ) };
  }
  template<typename ...Dependencies>
  __host__
  execute_with_dependencies<
    BaseSystem, cuda::unique_stream, Dependencies...
  >
  after(std::tuple<Dependencies...>& dependencies) const
  {
    return { std::tuple_cat(
      std::make_tuple(
        cuda::unique_stream(cuda::nonowning, stream)
      ),
      capture_as_dependency(dependencies)
    ) };
  }
  template<typename ...Dependencies>
  __host__
  execute_with_dependencies<
    BaseSystem, cuda::unique_stream, Dependencies...
  >
  after(std::tuple<Dependencies...>&& dependencies) const
  {
    return { std::tuple_cat(
      std::make_tuple(
        cuda::unique_stream(cuda::nonowning, stream)
      ),
      capture_as_dependency(std::move(dependencies))
    ) };
  }

  template<typename ...Dependencies>
  __host__
  execute_with_dependencies<
    BaseSystem, cuda::unique_stream, Dependencies...
  >
  rebind_after(Dependencies&& ...dependencies) const
  {
    return after(THRUST_FWD(dependencies)...);
  }
  template<typename ...Dependencies>
  __host__
  execute_with_dependencies<
    BaseSystem, cuda::unique_stream, Dependencies...
  >
  rebind_after(std::tuple<Dependencies...>& dependencies) const
  {
    return after(dependencies);
  }
  template<typename ...Dependencies>
  __host__
  execute_with_dependencies<
    BaseSystem, cuda::unique_stream, Dependencies...
  >
  rebind_after(std::tuple<Dependencies...>&& dependencies) const
  {
    return after(std::move(dependencies));
  }

  // This must be __host__ __device__, because `.on` has to work in device
  // code, although `.after` explicitly does not work in device code.
  friend __host__ __device__
  cudaStream_t
  dispatch_get_raw_stream(execute_with_stream const &system)
  {
    return system.stream;
  }

  friend __host__
  std::tuple<cuda::unique_stream>
  dispatch_extract_dependencies(execute_with_stream &system)
  {
    return { cuda::unique_stream(cuda::nonowning, system.stream) };
  }
};

template<typename Allocator, template<typename> class BaseSystem>
struct execute_with_allocator_and_stream
  : BaseSystem<execute_with_stream<BaseSystem>>
{
private:
  using super_t = BaseSystem<
    execute_with_allocator_and_stream<Allocator, BaseSystem>
  >;

  cudaStream_t const stream;
  Allocator alloc;

public:
  // This must be __host__ __device__, because `.on` has to work in device
  // code, although `.after` explicitly does not work in device code.
  __host__ __device__
  execute_with_allocator_and_stream(
      super_t const &super, Allocator alloc, cudaStream_t stream
    )
    : super_t(super), alloc(alloc), stream(stream)
  {}

  // This must be __host__ __device__, because `.on` has to work in device
  // code, although `.after` explicitly does not work in device code.
  __thrust_exec_check_disable__
  __host__ __device__
  execute_with_allocator_and_stream(
      Allocator alloc, cudaStream_t stream
    )
    : alloc(alloc), stream(stream)
  {}

  template<typename ...Dependencies>
  __host__
  execute_with_allocator_and_dependencies<
    Allocator, BaseSystem, cuda::unique_stream, Dependencies...
  >
  after(Dependencies&& ...dependencies) const
  {
    return { alloc, std::make_tuple(
      cuda::unique_stream(cuda::nonowning, stream),
      capture_as_dependency(THRUST_FWD(dependencies))...
    ) };
  }
  template<typename ...Dependencies>
  __host__
  execute_with_allocator_and_dependencies<
    Allocator, BaseSystem, cuda::unique_stream, Dependencies...
  >
  after(std::tuple<Dependencies...>& dependencies) const
  {
    return { alloc, std::tuple_cat(
      std::make_tuple(
        cuda::unique_stream(cuda::nonowning, stream)
      ),
      capture_as_dependency(dependencies)
    ) };
  }
  template<typename ...Dependencies>
  __host__
  execute_with_allocator_and_dependencies<
    Allocator, BaseSystem, cuda::unique_stream, Dependencies...
  >
  after(std::tuple<Dependencies...>&& dependencies) const
  {
    return { alloc, std::tuple_cat(
      std::make_tuple(
        cuda::unique_stream(cuda::nonowning, stream)
      ),
      capture_as_dependency(std::move(dependencies))
    ) };
  }

  template<typename ...Dependencies>
  __host__
  execute_with_allocator_and_dependencies<
    Allocator, BaseSystem, cuda::unique_stream, Dependencies...
  >
  rebind_after(Dependencies&& ...dependencies) const
  {
    return after(THRUST_FWD(dependencies)...);
  }
  template<typename ...Dependencies>
  __host__
  execute_with_allocator_and_dependencies<
    Allocator, BaseSystem, cuda::unique_stream, Dependencies...
  >
  rebind_after(std::tuple<Dependencies...>& dependencies) const
  {
    return after(dependencies);
  }
  template<typename ...Dependencies>
  __host__
  execute_with_allocator_and_dependencies<
    Allocator, BaseSystem, cuda::unique_stream, Dependencies...
  >
  rebind_after(std::tuple<Dependencies...>&& dependencies) const
  {
    return after(std::move(dependencies));
  }

  friend __host__
  typename std::add_lvalue_reference<Allocator>::type
  dispatch_get_allocator(execute_with_allocator_and_stream const &system)
  {
    return system.alloc;
  }

  // This must be __host__ __device__, because `.on` has to work in device
  // code, although `.after` explicitly does not work in device code.
  friend __host__ __device__
  cudaStream_t
  dispatch_get_raw_stream(execute_with_allocator_and_stream const &system)
  {
    return system.stream;
  }

  friend __host__
  std::tuple<cuda::unique_stream>
  dispatch_extract_dependencies(execute_with_allocator_and_stream &system)
  {
    return { cuda::unique_stream(cuda::nonowning, system.stream) };
  }
};

} // end detail
} // end thrust
