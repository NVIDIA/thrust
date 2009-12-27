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


/*! \file linear_congruential_engine.h
 *  \brief A linear congruential pseudorandom number engine.
 */

#pragma once

#include <thrust/detail/config.h>

namespace thrust
{

namespace experimental
{

namespace random
{

template<typename UIntType, UIntType a, UIntType c, UIntType m>
  class linear_congruential_engine
{
  public:
    // types
    typedef UIntType result_type;

    // engine characteristics
    static const result_type multiplier = a;
    static const result_type increment = c;
    static const result_type modulus = m;
    static const result_type min = c == 0u ? 1u : 0u;
    static const result_type max = m - 1u;
    static const result_type default_seed = 1u;

    // constructors and seeding functions
    __host__ __device__
    linear_congruential_engine(void);

    __host__ __device__
    explicit linear_congruential_engine(result_type s);

    __host__ __device__
    void seed(result_type s = default_seed);

    // generating functions
    __host__ __device__
    result_type operator()(void);

    __host__ __device__
    void discard(unsigned long long z);

    /*! \cond
     */
  private:
    result_type m_x;
    /*! \endcond
     */
}; // end linear_congruential

// XXX the type boost used here was boost::int32_t
typedef linear_congruential_engine<unsigned int, 16807, 0, 2147483647> minstd_rand0;
typedef linear_congruential_engine<unsigned int, 48271, 0, 2147483647> minstd_rand;

} // end experimental
  
} // end random

} // end thrust

#include <thrust/random/detail/linear_congruential_engine.inl>

