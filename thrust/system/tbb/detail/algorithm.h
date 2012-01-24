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

// the purpose of this header is to #include all the TBB
// backend entry point headers in tbb/detail

#include <thrust/system/tbb/detail/copy.h>
#include <thrust/system/tbb/detail/copy_if.h>
#include <thrust/system/tbb/detail/for_each.h>
#include <thrust/system/tbb/detail/merge.h>
#include <thrust/system/tbb/detail/reduce.h>
#include <thrust/system/tbb/detail/scan.h>
#include <thrust/system/tbb/detail/sort.h>

