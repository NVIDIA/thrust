/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

//functions to support blocking

namespace thrust
{

namespace detail
{

namespace util
{

// x/y rounding towards +infinity for integers, used to determine # of blocks/warps etc.
template<typename L, typename R>
  inline __host__ __device__ L divide_ri(const L x, const R y)
{
    return (x + (y - 1)) / y;
}

// x/y rounding towards zero for integers, used to determine # of blocks/warps etc.
template<typename L, typename R>
  inline __host__ __device__ L divide_rz(const L x, const R y)
{
    return x / y;
}

// round x towards infinity to the next multiple of y
template<typename L, typename R>
  inline __host__ __device__ L round_i(const L x, const R y){ return y * divide_ri(x, y); }

// round x towards zero to the next multiple of y
template<typename L, typename R>
  inline __host__ __device__ L round_z(const L x, const R y){ return y * divide_rz(x, y); }

} // end namespace util

} // end namespace detail

} // end namespace thrust

