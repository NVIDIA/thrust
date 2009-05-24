/*
 *  Copyright 2008-2009 NVIDIA Corporation
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

namespace warp
{

// XXX figure out the best interface

template <typename ValueType, typename BinaryFunction, unsigned int BLOCK_SIZE>
__device__ ValueType
inclusive_scan(ValueType * data, const unsigned int idx, BinaryFunction binary_op)
{
    const unsigned int lane = idx & 31;

    if (lane >=  1)  data[idx] = binary_op(data[idx -  1] , data[idx]);
    if (lane >=  2)  data[idx] = binary_op(data[idx -  2] , data[idx]);
    if (lane >=  4)  data[idx] = binary_op(data[idx -  4] , data[idx]);
    if (lane >=  8)  data[idx] = binary_op(data[idx -  8] , data[idx]);
    if (lane >= 16)  data[idx] = binary_op(data[idx - 16] , data[idx]);

    return data[idx];
}

} // end namespace warp

} // end namespace detail

} // end namespace thrust
