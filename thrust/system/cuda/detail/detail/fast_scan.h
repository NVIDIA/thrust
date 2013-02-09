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


/*! \file fast_scan.h
 *  \brief A fast scan for primitive types.
 */

#pragma once

namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace detail
{
namespace fast_scan
{

template <typename ExecutionPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename BinaryFunction>
OutputIterator inclusive_scan(execution_policy<ExecutionPolicy> &exec,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator output,
                              BinaryFunction binary_op);

template <typename ExecutionPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename T,
          typename BinaryFunction>
OutputIterator exclusive_scan(execution_policy<ExecutionPolicy> &exec,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator output,
                              const T init,
                              BinaryFunction binary_op);

} // end namespace fast_scan
} // end namespace detail
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

#include "fast_scan.inl"

