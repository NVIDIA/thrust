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

/*! \file subtract_with_carry_engine.h
 *  \brief A subtract-with-carry pseudorandom number generator
 *         based on Marsaglia & Zaman.
 */

#pragma once

#include <thrust/detail/config.h>

// XXX we should use <cstdint> instead
#include <stdint.h>
#include <iostream>

namespace thrust
{

namespace experimental
{

namespace random
{

template<typename UIntType, size_t w, size_t s, size_t r>
  class subtract_with_carry_engine
{
    /*! \cond
     */
  private:
    static const UIntType modulus = 1 << w;
    /*! \endcond
     */

  public:
    // types
    typedef UIntType result_type;

  public:

    // engine characteristics
    static const size_t word_size = w;
    static const size_t short_lag = s;
    static const size_t long_lag = r;
    static const result_type min = 0;
    static const result_type max = modulus - 1;
    static const result_type default_seed = 19780503u;

    // constructors and seeding functions
    __host__ __device__
    explicit subtract_with_carry_engine(result_type value = default_seed);

    __host__ __device__
    void seed(result_type value = default_seed);

    // generating functions
    __host__ __device__
    result_type operator()(void);

    __host__ __device__
    void discard(unsigned long long z);

    template<typename UIntType_, size_t w_, size_t s_, size_t r_,
             typename CharT, typename Traits>
    friend std::basic_ostream<CharT,Traits>&
    operator<<(std::basic_ostream<CharT,Traits> &os,
               const subtract_with_carry_engine<UIntType_,w_,s_,r_> &e);

    template<typename UIntType_, size_t w_, size_t s_, size_t r_,
             typename CharT, typename Traits>
    friend std::basic_istream<CharT,Traits>&
    operator>>(std::basic_istream<CharT,Traits> &is,
               subtract_with_carry_engine<UIntType_,w_,s_,r_> &e);

    /*! \cond
     */
  private:
    result_type m_x[long_lag];
    unsigned int m_k;
    int m_carry;
    /*! \endcond
     */
}; // end subtract_with_carry_engine


// XXX N2111 uses uint_fast32_t here
typedef subtract_with_carry_engine<uint32_t, 24, 10, 24> ranlux24_base;

// XXX N2111 uses uint_fast64_t here
typedef subtract_with_carry_engine<uint64_t, 48,  5, 12> ranlux48_base;

} // end random

} // end experimental

} // end thrust

#include <thrust/random/detail/subtract_with_carry_engine.inl>

