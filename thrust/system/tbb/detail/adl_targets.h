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

// the purpose of this header is to #include all of tbb::tag's
// ADL targets

#include <thrust/system/tbb/detail/adjacent_difference.h>
#include <thrust/system/tbb/detail/assign_value.h>
#include <thrust/system/tbb/detail/binary_search.h>
#include <thrust/system/tbb/detail/copy.h>
#include <thrust/system/tbb/detail/copy_if.h>
#include <thrust/system/tbb/detail/count.h>
#include <thrust/system/tbb/detail/equal.h>
#include <thrust/system/tbb/detail/extrema.h>
#include <thrust/system/tbb/detail/fill.h>
#include <thrust/system/tbb/detail/find.h>
#include <thrust/system/tbb/detail/for_each.h>
#include <thrust/system/tbb/detail/gather.h>
#include <thrust/system/tbb/detail/generate.h>
#include <thrust/system/tbb/detail/get_value.h>
#include <thrust/system/tbb/detail/inner_product.h>
#include <thrust/system/tbb/detail/iter_swap.h>
#include <thrust/system/tbb/detail/logical.h>
#include <thrust/system/tbb/detail/malloc_and_free.h>
#include <thrust/system/tbb/detail/merge.h>
#include <thrust/system/tbb/detail/mismatch.h>
#include <thrust/system/tbb/detail/partition.h>
#include <thrust/system/tbb/detail/reduce.h>
#include <thrust/system/tbb/detail/reduce_by_key.h>
#include <thrust/system/tbb/detail/remove.h>
#include <thrust/system/tbb/detail/replace.h>
#include <thrust/system/tbb/detail/reverse.h>
#include <thrust/system/tbb/detail/scan.h>
#include <thrust/system/tbb/detail/scan_by_key.h>
#include <thrust/system/tbb/detail/scatter.h>
#include <thrust/system/tbb/detail/sequence.h>
#include <thrust/system/tbb/detail/set_operations.h>
#include <thrust/system/tbb/detail/sort.h>
#include <thrust/system/tbb/detail/swap_ranges.h>
#include <thrust/system/tbb/detail/tabulate.h>
#include <thrust/system/tbb/detail/transform.h>
#include <thrust/system/tbb/detail/transform_reduce.h>
#include <thrust/system/tbb/detail/transform_scan.h>
#include <thrust/system/tbb/detail/uninitialized_copy.h>
#include <thrust/system/tbb/detail/uninitialized_fill.h>
#include <thrust/system/tbb/detail/unique.h>
#include <thrust/system/tbb/detail/unique_by_key.h>

