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
#include <thrust/detail/cpp11_required.h>

#if THRUST_CPP_DIALECT >= 2011

#include <tuple>

namespace thrust
{
namespace detail
{

template<template<typename> class BaseSystem, typename... Dependencies>
struct execute_with_dependencies
    : BaseSystem<execute_with_dependencies<BaseSystem, Dependencies...>>
{
private:
    using super_t = BaseSystem<execute_with_dependencies<BaseSystem, Dependencies...>>;

    std::tuple<Dependencies...> dependencies;

public:
    __host__
    execute_with_dependencies(super_t const &super, Dependencies && ...dependencies)
        : super_t(super), dependencies(std::forward<Dependencies>(dependencies)...)
    {
    }

    __host__
    execute_with_dependencies(Dependencies && ...dependencies)
        : dependencies(std::forward<Dependencies>(dependencies)...)
    {
    }

    std::tuple<Dependencies...>
    __host__
    extract_dependencies() &&
    {
        return std::move(dependencies);
    }
};

template<
    typename Allocator,
    template<typename> class BaseSystem,
    typename... Dependencies
>
struct execute_with_allocator_and_dependencies
    : BaseSystem<
        execute_with_allocator_and_dependencies<
            Allocator,
            BaseSystem,
            Dependencies...
        >
    >
{
private:
    using super_t = BaseSystem<
        execute_with_allocator_and_dependencies<
            Allocator,
            BaseSystem,
            Dependencies...
        >
    >;

    std::tuple<Dependencies...> dependencies;
    Allocator alloc;

public:
    __host__
    execute_with_allocator_and_dependencies(super_t const &super, Allocator alloc, Dependencies && ...dependencies)
        : super_t(super), alloc(alloc), dependencies(std::forward<Dependencies>(dependencies)...)
    {
    }

    __host__
    execute_with_allocator_and_dependencies(Allocator alloc, Dependencies && ...dependencies)
        : alloc(alloc), dependencies(std::forward<Dependencies>(dependencies)...)
    {
    }

    std::tuple<Dependencies...>
    __host__
    extract_dependencies() &&
    {
        return std::move(dependencies);
    }

    Allocator
    __host__
    get_allocator()
    {
        return alloc;
    }
};

template<template<typename> class BaseSystem, typename ...Dependencies>
__host__
std::tuple<Dependencies...>
extract_dependencies(thrust::detail::execute_with_dependencies<BaseSystem, Dependencies...>&& system)
{
    return std::move(system).extract_dependencies();
}

template<typename Allocator, template<typename> class BaseSystem, typename ...Dependencies>
__host__
std::tuple<Dependencies...>
extract_dependencies(thrust::detail::execute_with_allocator_and_dependencies<Allocator, BaseSystem, Dependencies...>&& system)
{
    return std::move(system).extract_dependencies();
}

template<typename System>
__host__
std::tuple<>
extract_dependencies(System &&)
{
    return std::tuple<>{};
}

} // end detail
} // end thrust

#endif // THRUST_CPP_DIALECT >= 2011

