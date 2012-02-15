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

#include <thrust/detail/type_traits.h>

#include <thrust/detail/backend/cuda/detail/fast_scan.h>
#include <thrust/detail/backend/cuda/detail/safe_scan.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace cuda
{
namespace dispatch
{

/////////////////////
// Fast Scan Paths //
/////////////////////

template<typename InputIterator,
         typename OutputIterator,
         typename AssociativeOperator>
  OutputIterator inclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                AssociativeOperator binary_op,
                                thrust::detail::true_type)    // use fast_scan
{
    return thrust::detail::backend::cuda::detail::fast_scan::inclusive_scan
        (first, last, result, binary_op);
}

template<typename InputIterator,
         typename OutputIterator,
         typename T,
         typename AssociativeOperator>
  OutputIterator exclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T init,
                                AssociativeOperator binary_op,
                                thrust::detail::true_type)    // use fast_scan
{
    return thrust::detail::backend::cuda::detail::fast_scan::exclusive_scan
        (first, last, result, init, binary_op);
}

/////////////////////
// Safe Scan Paths //
/////////////////////

template<typename InputIterator,
         typename OutputIterator,
         typename AssociativeOperator>
  OutputIterator inclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                AssociativeOperator binary_op,
                                thrust::detail::false_type)    // use safe_scan
{
    return thrust::detail::backend::cuda::detail::safe_scan::inclusive_scan
        (first, last, result, binary_op);
}

template<typename InputIterator,
         typename OutputIterator,
         typename T,
         typename AssociativeOperator>
  OutputIterator exclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T init,
                                AssociativeOperator binary_op,
                                thrust::detail::false_type)    // use safe_scan
{
    return thrust::detail::backend::cuda::detail::safe_scan::exclusive_scan
        (first, last, result, init, binary_op);
}

} // end namespace dispatch
} // end namespace cuda
} // end namespace backend
} // end namespace detail
} // end namespace thrust

