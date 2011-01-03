/*
 *  Copyright 2008-2011 NVIDIA Corporation
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

/*! \file device_backend.h
 *  \brief Device backend-specific configuration.
 */

#pragma once

// XXX reserve 0 for undefined
#define THRUST_DEVICE_BACKEND_CUDA    1
#define THRUST_DEVICE_BACKEND_OMP     2

#ifndef THRUST_DEVICE_BACKEND
#define THRUST_DEVICE_BACKEND THRUST_DEVICE_BACKEND_CUDA
#endif // THRUST_DEVICE_BACKEND

#if defined(__DEVICE_EMULATION__)
#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
#pragma message("-----------------------------------------------------------------------")
#pragma message("| WARNING: Thrust does not support device emulation                    ")
#pragma message("-----------------------------------------------------------------------")
#else
#warning -----------------------------------------------------------------------
#warning | WARNING: Thrust does not support device emulation                    
#warning -----------------------------------------------------------------------
#endif // THRUST_HOST_COMPILER_MSVC
#endif // __DEVICE_EMULATION__

