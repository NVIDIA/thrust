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
#include <thrust/system/detail/sequential/tag.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace sequential
{

#ifdef __CUDA_ARCH__
static const __device__ tag seq;
#else
static const tag seq;
#endif

}
}
}
}

#include <thrust/system/detail/sequential/adjacent_difference.h>
#include <thrust/system/detail/sequential/binary_search.h>
#include <thrust/system/detail/sequential/copy_if.h>
#include <thrust/system/detail/sequential/extrema.h>
#include <thrust/system/detail/sequential/find.h>
#include <thrust/system/detail/sequential/for_each.h>
#include <thrust/system/detail/sequential/reduce_by_key.h>
#include <thrust/system/detail/sequential/reduce.h>
#include <thrust/system/detail/sequential/remove.h>
#include <thrust/system/detail/sequential/scan_by_key.h>

