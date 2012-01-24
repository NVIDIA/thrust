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

// the purpose of this header is to #include all the CUDA
// backend entry point headers in cuda/detail

// the order of the following #includes seems to matter, unfortunately

// primitives come first, in order of increasing sophistication
#include <thrust/system/cuda/detail/for_each.h>
#include <thrust/system/cuda/detail/copy.h>
#include <thrust/system/cuda/detail/reduce.h>
#include <thrust/system/cuda/detail/scan.h>
#include <thrust/system/cuda/detail/sort.h>

// these are alphabetical
#include <thrust/system/cuda/detail/adjacent_difference.h>
#include <thrust/system/cuda/detail/copy_if.h>
#include <thrust/system/cuda/detail/fill.h>
#include <thrust/system/cuda/detail/merge.h>
#include <thrust/system/cuda/detail/reduce_by_key.h>
#include <thrust/system/cuda/detail/set_operations.h>

