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

// the purpose of this header is to #include all of omp::tag's
// ADL targets

#include <thrust/system/omp/detail/adjacent_difference.h>
#include <thrust/system/omp/detail/assign_value.h>
#include <thrust/system/omp/detail/binary_search.h>
#include <thrust/system/omp/detail/copy.h>
#include <thrust/system/omp/detail/copy_if.h>
#include <thrust/system/omp/detail/count.h>
#include <thrust/system/omp/detail/equal.h>
#include <thrust/system/omp/detail/extrema.h>
#include <thrust/system/omp/detail/fill.h>
#include <thrust/system/omp/detail/find.h>
#include <thrust/system/omp/detail/for_each.h>
#include <thrust/system/omp/detail/gather.h>
#include <thrust/system/omp/detail/generate.h>
#include <thrust/system/omp/detail/get_value.h>
#include <thrust/system/omp/detail/inner_product.h>
#include <thrust/system/omp/detail/iter_swap.h>
#include <thrust/system/omp/detail/logical.h>
#include <thrust/system/omp/detail/malloc_and_free.h>
#include <thrust/system/omp/detail/merge.h>
#include <thrust/system/omp/detail/mismatch.h>
#include <thrust/system/omp/detail/partition.h>
#include <thrust/system/omp/detail/reduce.h>
#include <thrust/system/omp/detail/reduce_by_key.h>
#include <thrust/system/omp/detail/remove.h>
#include <thrust/system/omp/detail/replace.h>
#include <thrust/system/omp/detail/reverse.h>
#include <thrust/system/omp/detail/scan.h>
#include <thrust/system/omp/detail/scan_by_key.h>
#include <thrust/system/omp/detail/scatter.h>
#include <thrust/system/omp/detail/sequence.h>
#include <thrust/system/omp/detail/set_operations.h>
#include <thrust/system/omp/detail/sort.h>
#include <thrust/system/omp/detail/swap_ranges.h>
#include <thrust/system/omp/detail/tabulate.h>
#include <thrust/system/omp/detail/transform.h>
#include <thrust/system/omp/detail/transform_reduce.h>
#include <thrust/system/omp/detail/transform_scan.h>
#include <thrust/system/omp/detail/uninitialized_copy.h>
#include <thrust/system/omp/detail/uninitialized_fill.h>
#include <thrust/system/omp/detail/unique.h>
#include <thrust/system/omp/detail/unique_by_key.h>

