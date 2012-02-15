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

namespace thrust
{

namespace detail
{

namespace backend
{

namespace cuda
{

namespace warp
{

template<typename InputType, 
         typename InputIterator, 
         typename AssociativeOperator>
         __device__
InputType scan(const unsigned int thread_lane, InputType val, InputIterator sdata, AssociativeOperator binary_op)
{
    sdata[threadIdx.x] = val;

    if (thread_lane >=  1)  sdata[threadIdx.x] = val = binary_op(sdata[threadIdx.x -  1], val);
    if (thread_lane >=  2)  sdata[threadIdx.x] = val = binary_op(sdata[threadIdx.x -  2], val);
    if (thread_lane >=  4)  sdata[threadIdx.x] = val = binary_op(sdata[threadIdx.x -  4], val);
    if (thread_lane >=  8)  sdata[threadIdx.x] = val = binary_op(sdata[threadIdx.x -  8], val);
    if (thread_lane >= 16)  sdata[threadIdx.x] = val = binary_op(sdata[threadIdx.x - 16], val);

    return val;
}

} // end namespace warp

} // end namespace cuda

} // end namespace backend

} // end namespace detail

} // end namespace thrust

