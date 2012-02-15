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

/*! \file host_device.h
 *  \brief Defines __host__ and __device__ and other CUDA-isms
 */

#pragma once

#include <thrust/detail/config.h>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

#include <host_defines.h>

#else

// since __host__ & __device__ might have already be defined, only
// #define them if not defined already
// XXX this will break if the client does #include <host_defines.h> later

#ifndef __host__
#define __host__
#endif // __host__

#ifndef __device__
#define __device__
#endif // __device__

#endif

