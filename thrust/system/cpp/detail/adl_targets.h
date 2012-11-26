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

// the purpose of this header is to #include all of cpp::tag's
// ADL targets

#include <thrust/system/cpp/detail/adjacent_difference.h>
#include <thrust/system/cpp/detail/assign_value.h>
#include <thrust/system/cpp/detail/binary_search.h>
#include <thrust/system/cpp/detail/copy.h>
#include <thrust/system/cpp/detail/copy_if.h>
#include <thrust/system/cpp/detail/count.h>
#include <thrust/system/cpp/detail/equal.h>
#include <thrust/system/cpp/detail/extrema.h>
#include <thrust/system/cpp/detail/fill.h>
#include <thrust/system/cpp/detail/find.h>
#include <thrust/system/cpp/detail/for_each.h>
#include <thrust/system/cpp/detail/gather.h>
#include <thrust/system/cpp/detail/generate.h>
#include <thrust/system/cpp/detail/get_value.h>
#include <thrust/system/cpp/detail/inner_product.h>
#include <thrust/system/cpp/detail/iter_swap.h>
#include <thrust/system/cpp/detail/logical.h>
#include <thrust/system/cpp/detail/malloc_and_free.h>
#include <thrust/system/cpp/detail/merge.h>
#include <thrust/system/cpp/detail/mismatch.h>
#include <thrust/system/cpp/detail/partition.h>
#include <thrust/system/cpp/detail/reduce.h>
#include <thrust/system/cpp/detail/reduce_by_key.h>
#include <thrust/system/cpp/detail/remove.h>
#include <thrust/system/cpp/detail/replace.h>
#include <thrust/system/cpp/detail/reverse.h>
#include <thrust/system/cpp/detail/scan.h>
#include <thrust/system/cpp/detail/scan_by_key.h>
#include <thrust/system/cpp/detail/scatter.h>
#include <thrust/system/cpp/detail/sequence.h>
#include <thrust/system/cpp/detail/set_operations.h>
#include <thrust/system/cpp/detail/sort.h>
#include <thrust/system/cpp/detail/swap_ranges.h>
#include <thrust/system/cpp/detail/tabulate.h>
#include <thrust/system/cpp/detail/transform.h>
#include <thrust/system/cpp/detail/transform_reduce.h>
#include <thrust/system/cpp/detail/transform_scan.h>
#include <thrust/system/cpp/detail/uninitialized_copy.h>
#include <thrust/system/cpp/detail/uninitialized_fill.h>
#include <thrust/system/cpp/detail/unique.h>
#include <thrust/system/cpp/detail/unique_by_key.h>

