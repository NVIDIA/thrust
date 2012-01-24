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

// the purpose of this header is to #include all the cpp
// backend entry point headers in cpp/detail

#include <thrust/system/cpp/detail/adjacent_difference.h>
#include <thrust/system/cpp/detail/binary_search.h>
#include <thrust/system/cpp/detail/copy.h>
#include <thrust/system/cpp/detail/extrema.h>
#include <thrust/system/cpp/detail/find.h>
#include <thrust/system/cpp/detail/for_each.h>
#include <thrust/system/cpp/detail/merge.h>
#include <thrust/system/cpp/detail/partition.h>
#include <thrust/system/cpp/detail/reduce.h>
#include <thrust/system/cpp/detail/reduce_by_key.h>
#include <thrust/system/cpp/detail/remove.h>
#include <thrust/system/cpp/detail/scan.h>
#include <thrust/system/cpp/detail/scan_by_key.h>
#include <thrust/system/cpp/detail/set_operations.h>
#include <thrust/system/cpp/detail/sort.h>
#include <thrust/system/cpp/detail/unique.h>
#include <thrust/system/cpp/detail/unique_by_key.h>

