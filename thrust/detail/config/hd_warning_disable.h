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

/*! \file hd_warning_disable.h
 *  \brief Defines __thrust_hd_warning_disable__
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(__CUDACC__)

#define __thrust_hd_warning_disable__ \
#pragma hd_warning_disable
#else

#define __thrust_hd_warning_disable__

#endif


