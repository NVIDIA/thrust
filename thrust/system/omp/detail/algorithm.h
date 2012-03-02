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

// the purpose of this header is to #include all the omp
// backend entry point headers in omp/detail

#include <thrust/system/omp/detail/adjacent_difference.h>
#include <thrust/system/omp/detail/binary_search.h>
#include <thrust/system/omp/detail/copy.h>
#include <thrust/system/omp/detail/copy_if.h>
#include <thrust/system/omp/detail/extrema.h>
#include <thrust/system/omp/detail/find.h>
#include <thrust/system/omp/detail/for_each.h>
#include <thrust/system/omp/detail/partition.h>
#include <thrust/system/omp/detail/reduce.h>
#include <thrust/system/omp/detail/reduce_intervals.h>
#include <thrust/system/omp/detail/reduce_by_key.h>
#include <thrust/system/omp/detail/remove.h>
#include <thrust/system/omp/detail/sort.h>
#include <thrust/system/omp/detail/unique.h>
#include <thrust/system/omp/detail/unique_by_key.h>

