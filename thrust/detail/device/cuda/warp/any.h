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

namespace warp
{

template<typename InputType, 
         typename InputIterator>
         __device__
InputType any(const unsigned int thread_lane, InputType val, InputIterator sdata)
{
#if __CUDA_ARCH__ >= 120
    // optimization, use __any(cond)
    return __any(val);
#else
    sdata[threadIdx.x] = val;

    if (thread_lane >=  1)  sdata[threadIdx.x] = val = val | sdata[threadIdx.x -  1];
    if (thread_lane >=  2)  sdata[threadIdx.x] = val = val | sdata[threadIdx.x -  2];
    if (thread_lane >=  4)  sdata[threadIdx.x] = val = val | sdata[threadIdx.x -  4];
    if (thread_lane >=  8)  sdata[threadIdx.x] = val = val | sdata[threadIdx.x -  8];
    if (thread_lane >= 16)  sdata[threadIdx.x] = val = val | sdata[threadIdx.x - 16];

    return sdata[threadIdx.x - thread_lane + 32];
#endif
}

} // end namespace warp

} // end namespace cuda

} // end namespace device

} // end namespace detail

} // end namespace thrust

