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

/*! \file execution_policy.h
 *  \brief Thrust execution policies.
 */

#pragma once

#include <thrust/detail/config.h>

// get the definition of thrust::execution_policy
#include <thrust/detail/execution_policy.h>

// #include the host system's execution_policy header
#define __THRUST_HOST_SYSTEM_EXECUTION_POLICY_HEADER <__THRUST_HOST_SYSTEM_ROOT/execution_policy.h>
#include __THRUST_HOST_SYSTEM_EXECUTION_POLICY_HEADER
#undef __THRUST_HOST_SYSTEM_EXECUTION_POLICY_HEADER

// #include the device system's execution_policy.h header
#define __THRUST_DEVICE_SYSTEM_EXECUTION_POLICY_HEADER <__THRUST_DEVICE_SYSTEM_ROOT/execution_policy.h>
#include __THRUST_DEVICE_SYSTEM_EXECUTION_POLICY_HEADER
#undef __THRUST_DEVICE_SYSTEM_EXECUTION_POLICY_HEADER

namespace thrust
{
namespace detail
{

typedef thrust::system::__THRUST_HOST_SYSTEM_NAMESPACE::detail::par_t host_t;

typedef thrust::system::__THRUST_DEVICE_SYSTEM_NAMESPACE::detail::par_t device_t;

} // end detail

template<typename DerivedPolicy>
  struct host_execution_policy
    : thrust::system::__THRUST_HOST_SYSTEM_NAMESPACE::execution_policy<DerivedPolicy>
{};

template<typename DerivedPolicy>
  struct device_execution_policy
    : thrust::system::__THRUST_DEVICE_SYSTEM_NAMESPACE::execution_policy<DerivedPolicy>
{};

static const detail::host_t host;

static const detail::device_t device;

} // end thrust

