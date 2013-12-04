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

#include <thrust/detail/config.h>
#include <limits>


namespace thrust
{
namespace detail
{


template<typename Integer>
__host__ __device__ __thrust_forceinline__
Integer clz(Integer x)
{
  // XXX optimize by lowering to intrinsics
  
  int num_bits = 8 * sizeof(Integer);
  int num_bits_minus_one = num_bits - 1;

  for(int i = num_bits_minus_one; i >= 0; --i)
  {
    if((Integer(1) << i) & x)
    {
      return num_bits_minus_one - i;
    }
  }

  return num_bits;
}


template<typename Integer>
__host__ __device__ __thrust_forceinline__
bool is_power_of_2(Integer x)
{
  return 0 == (x & (x - 1));
}


template<typename Integer>
__host__ __device__ __thrust_forceinline__
Integer log2(Integer x)
{
  Integer num_bits = 8 * sizeof(Integer);
  Integer num_bits_minus_one = num_bits - 1;

  return num_bits_minus_one - clz(x);
}


template<typename Integer>
__host__ __device__ __thrust_forceinline__
Integer log2_ri(Integer x)
{
  Integer result = log2(x);

  // this is where we round up to the nearest log
  if(!is_power_of_2(x))
  {
    ++result;
  }

  return result;
}


template<typename Integer>
__host__ __device__ __thrust_forceinline__
bool is_odd(Integer x)
{
  return 1 & x;
}


} // end detail
} // end thrust

