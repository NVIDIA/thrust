/*
 *  Copyright 2008-2011 NVIDIA Corporation
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

namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{

/* Split the sequence [0,N) into uniformly-sized sub-intervals.
 *
 * Returns a pair of integers
 *  (interval_size, num_intervals)
 * such that
 *  - interval_size is a positive multiple of granularity
 *  - num_intervals is in the range [1,max_intervals]
 *  - the size of the last interval is in the range [1,interval_size]
 *
 */
template <typename T>
thrust::pair<T,T> uniform_interval_splitting(const T N, const T granularity, const T max_intervals)
{
    const T grains  = thrust::detail::util::divide_ri(N, granularity);

    // one grain per interval
    if (grains <= max_intervals)
        return thrust::make_pair(granularity, grains);

    // insures that:
    //     num_intervals * interval_size is >= N 
    //   and 
    //     (num_intervals - 1) * interval_size is < N
    const T grains_per_interval = thrust::detail::util::divide_ri(grains, max_intervals);
    const T interval_size       = grains_per_interval * granularity;
    const T num_intervals       = thrust::detail::util::divide_ri(N, interval_size);

    return thrust::make_pair(interval_size, num_intervals);
}

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

