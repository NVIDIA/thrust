/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

/*! \file thrust/future.h
 *  \brief Thrust's asynchronous handle.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp11_required.h>

#if THRUST_CPP_DIALECT >= 2011

#include <thrust/execution_policy.h>

// #include the device system's pointer.h header.
#define __THRUST_DEVICE_SYSTEM_POINTER_HEADER <__THRUST_DEVICE_SYSTEM_ROOT/pointer.h>
  #include __THRUST_DEVICE_SYSTEM_POINTER_HEADER
#undef __THRUST_DEVICE_SYSTEM_POINTER_HEADER

//// #include the host system's pointer.h header.
//#define __THRUST_HOST_SYSTEM_POINTER_HEADER <__THRUST_HOST_SYSTEM_ROOT/pointer.h>
//  #include __THRUST_HOST_SYSTEM_POINTER_HEADER
//#undef __THRUST_HOST_SYSTEM_POINTER_HEADER

THRUST_BEGIN_NS

// Fallback.
template <typename T, typename Pointer>
void unique_eager_future_type(...);

template <
  typename T
, typename System = thrust::system::__THRUST_DEVICE_SYSTEM_NAMESPACE::tag
, typename Pointer = thrust::system::__THRUST_DEVICE_SYSTEM_NAMESPACE::pointer<T>
>
  using unique_eager_future = decltype(unique_eager_future_type<T, Pointer>(
    std::declval<System>()
  ));
template <
  typename T
, typename System = thrust::system::__THRUST_DEVICE_SYSTEM_NAMESPACE::tag
, typename Pointer = thrust::system::__THRUST_DEVICE_SYSTEM_NAMESPACE::pointer<T>
>
  using future = unique_eager_future<T, System, Pointer>;

//template <
//  typename T
//, typename Pointer = thrust::system::__THRUST_HOST_SYSTEM_NAMESPACE::pointer<T>
//>
//  using host_unique_eager_future
//    = decltype(unique_eager_future_type<T, Pointer>(
//        std::declval<thrust::system::__THRUST_HOST_SYSTEM_NAMESPACE::tag>()
//      ));
//template <
//  typename T
//, typename Pointer = thrust::system::__THRUST_HOST_SYSTEM_NAMESPACE::pointer<T>
//>
//  using host_future = host_unique_eager_future<T>;

template <
  typename T
, typename Pointer = thrust::system::__THRUST_DEVICE_SYSTEM_NAMESPACE::pointer<T>
>
  using device_unique_eager_future
    = decltype(unique_eager_future_type<T, Pointer>(
        std::declval<thrust::system::__THRUST_DEVICE_SYSTEM_NAMESPACE::tag>()
      ));
template <
  typename T
, typename Pointer = thrust::system::__THRUST_DEVICE_SYSTEM_NAMESPACE::pointer<T>
>
  using device_future = device_unique_eager_future<T, Pointer>;

THRUST_END_NS

// #include the device system's future.h header.
#define __THRUST_DEVICE_SYSTEM_FUTURE_HEADER <__THRUST_DEVICE_SYSTEM_ROOT/future.h>
  #include __THRUST_DEVICE_SYSTEM_FUTURE_HEADER
#undef __THRUST_DEVICE_SYSTEM_FUTURE_HEADER

//// #include the host system's future.h header.
//#define __THRUST_HOST_SYSTEM_FUTURE_HEADER <__THRUST_HOST_SYSTEM_ROOT/future.h>
//  #include __THRUST_HOST_SYSTEM_FUTURE_HEADER
//#undef __THRUST_HOST_SYSTEM_FUTURE_HEADER

#endif // THRUST_CPP_DIALECT >= 2011

