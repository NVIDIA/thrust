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


/*! \file linear_congruential.h
 *  \brief A linear congruential pseudorandom number generator.
 */

#pragma once

namespace thrust
{

namespace experimental
{

namespace random
{

template<typename UIntType, UIntType a, UIntType c, UIntType m, UIntType val>
  class linear_congruential
{
  public:
    typedef UIntType result_type;

    static const UIntType multiplier = a;
    static const UIntType increment  = c;

    // XXX check m for 0. in that case,
    // modulus = std::numeric_limits<UIntType>::max()
    static const UIntType modulus    = m;

    __host__ __device__
    explicit linear_congruential(unsigned long x0 = 1);

    template<typename Gen>
    __host__ __device__
    linear_congruential(Gen &g);

    __host__ __device__
    void seed(unsigned long x0 = 1);

    template<typename Gen>
    __host__ __device__
    void seed(Gen &g);

    __host__ __device__
    result_type min(void) const;

    __host__ __device__
    result_type max(void) const;

    __host__ __device__
    result_type operator()(void);

    /*! \cond
     */
  private:
    UIntType m_x;

    static const result_type min_value = (c == 0 ? 1 : 0);
    static const result_type max_value = m-1;
    /*! \endcond
     */
}; // end linear_congruential

// XXX the type boost used here was boost::int32_t
typedef linear_congruential<int, 16807, 0, 2147483647> minstd_rand0;
typedef linear_congruential<int, 48271, 0, 2147483647> minstd_rand;

} // end experimental
  
} // end random

} // end thrust

#include <thrust/random/detail/linear_congruential.inl>

