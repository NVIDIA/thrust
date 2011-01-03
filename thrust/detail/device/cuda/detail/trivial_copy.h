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

namespace detail
{

enum trivial_copy_kind
{
  trivial_copy_warp,
  trivial_copy_block,
  trivial_copy_base
};

template<trivial_copy_kind kind>
__device__ void trivial_copy(void* _destination, const void* _source, size_t length)
{
  int index;
  int stride;
  
  char* destination  = reinterpret_cast<char*>(_destination);
  const char* source = reinterpret_cast<const char*>(_source);
  
  switch(kind)
  {
    case trivial_copy_warp:
    {
      index  = threadIdx.x % warpSize;
      stride = warpSize;
      break;
    }
    case trivial_copy_block:
    {
      index  = threadIdx.x;
      stride = blockDim.x;
      break;
    }
    case trivial_copy_base:
    {
      index = threadIdx.x + blockIdx.x * gridDim.x;
      stride = blockDim.x * gridDim.x;
      break;
    }
  }
  
  // check alignment
  // XXX can we do this in three steps?
  //     1. copy until alignment is met
  //     2. go hog wild
  //     3. get the remainder
  if(reinterpret_cast<size_t>(destination) % sizeof(uint2) != 0 || reinterpret_cast<size_t>(source) % sizeof(uint2) != 0)
  {
    for(unsigned int i = index; i < length; i += stride)
    {
      destination[i] = source[i];
    }
  }
  else
  {
    // it's aligned; go hog wild
    int steps = length/sizeof(int2);
    int double_stride = stride * 2;
    int i;

    const int2 *src_wide = reinterpret_cast<const int2*>(source);
          int2 *dst_wide = reinterpret_cast<int2*>(destination);
    
    // transfer bulk
    for(i = 0; i < steps - double_stride; i += double_stride)
    {
      int2 tempA = src_wide[i + index + stride * 0];
      int2 tempB = src_wide[i + index + stride * 1];
      
      dst_wide[i + index + stride * 0] = tempA;
      dst_wide[i + index + stride * 1] = tempB;
    }
    
    // transfer remainder
    for(; i < steps; i += stride)
    {
      if((i + index) < steps)
      {
        dst_wide[i + index] = src_wide[i + index];
      }
    }
    
    // transfer last few bytes
    for(i = length - length % sizeof(int2); i < length; i++)
    {
      destination[i] = source[i];
    }
  }
}

} // end detail

} // end cuda

} // end device

} // end detail

} // end thrust

