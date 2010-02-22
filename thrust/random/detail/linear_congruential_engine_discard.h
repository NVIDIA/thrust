/*
 *  Copyright 2008-2010 NVIDIA Corporation
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

#include <thrust/detail/cstdint.h>

namespace thrust
{

namespace random
{

namespace detail
{


template<typename UIntType, unsigned long long c>
  struct linear_congruential_engine_discard
{
  template<typename LinearCongruentialEngine>
  __host__ __device__
  static void discard(LinearCongruentialEngine &lcg, unsigned long long z)
  {
    for(; z > 0; --z)
    {
      lcg();
    }
  }
}; // end linear_congruential_engine_discard


// specialize for small integers and c == 0
// XXX figure out a robust implemenation of this for any unsigned integer type later
template<>
  struct linear_congruential_engine_discard<thrust::detail::uint32_t,0>
{
  template<typename T, T m>
  __host__ __device__
  static T mult_mod(T a, T x)
  {
    if(a == 1)
    {
      x %= m;
    }
    else
    {
      const T q = m / a;
      const T r = m % a;

      const T t1 = a * (x % q);
      const T t2 = r * (x / q);
      if(t1 >= t2)
      {
        x = t1 - t2;
      }
      else
      {
        x = m - t2 + t1;
      }
    }

    return x;
  }


  template<typename LinearCongruentialEngine>
  __host__ __device__
  static void discard(LinearCongruentialEngine &lcg, unsigned long long z)
  {
    typedef typename LinearCongruentialEngine::result_type result_type;
    const result_type modulus = LinearCongruentialEngine::modulus;

    // XXX we need to use unsigned long long here or we will encounter overflow in the
    //     multiplies below
    //     figure out a robust implementation of this later
    unsigned long long multiplier = LinearCongruentialEngine::multiplier;
    unsigned long long multiplier_to_z = 1;
    
    // see http://en.wikipedia.org/wiki/Modular_exponentiation
    while(z > 0)
    {
      if(z & 1)
      {
        // multiply in this bit's contribution while using modulus to keep result small
        multiplier_to_z = (multiplier_to_z * multiplier) % modulus;
      }

      // move to the next bit of the exponent, square (and mod) the base accordingly
      z >>= 1;
      multiplier = (multiplier * multiplier) % modulus;
    }

    lcg.m_x = (multiplier_to_z * lcg.m_x) % modulus;
  }
}; // end linear_congruential_engine_discard


} // end detail

} // end random

} // end thrust

