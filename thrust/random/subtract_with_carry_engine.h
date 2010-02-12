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

/*! \file subtract_with_carry_engine.h
 *  \brief A subtract-with-carry pseudorandom number generator
 *         based on Marsaglia & Zaman.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/random/detail/random_core_access.h>

#include <thrust/detail/cstdint.h>
#include <cstddef> // for size_t
#include <iostream>

namespace thrust
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

    /*! \cond
     */
  private:
    result_type m_x[long_lag];
    unsigned int m_k;
    int m_carry;

    friend struct thrust::random::detail::random_core_access;

    __host__ __device__
    bool equal(const subtract_with_carry_engine &rhs) const;

    template<typename CharT, typename Traits>
    std::basic_ostream<CharT,Traits>& stream_out(std::basic_ostream<CharT,Traits> &os) const;

    template<typename CharT, typename Traits>
    std::basic_istream<CharT,Traits>& stream_in(std::basic_istream<CharT,Traits> &is);

    /*! \endcond
     */
}; // end subtract_with_carry_engine


template<typename UIntType_, size_t w_, size_t s_, size_t r_>
__host__ __device__
bool operator==(const subtract_with_carry_engine<UIntType_,w_,s_,r_> &lhs,
                const subtract_with_carry_engine<UIntType_,w_,s_,r_> &rhs);

template<typename UIntType_, size_t w_, size_t s_, size_t r_>
__host__ __device__
bool operator!=(const subtract_with_carry_engine<UIntType_,w_,s_,r_>&lhs,
                const subtract_with_carry_engine<UIntType_,w_,s_,r_>&rhs);

template<typename UIntType_, size_t w_, size_t s_, size_t r_,
         typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os,
           const subtract_with_carry_engine<UIntType_,w_,s_,r_> &e);

template<typename UIntType_, size_t w_, size_t s_, size_t r_,
         typename CharT, typename Traits>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits> &is,
           subtract_with_carry_engine<UIntType_,w_,s_,r_> &e);


// XXX N2111 uses uint_fast32_t here
typedef subtract_with_carry_engine<thrust::detail::uint32_t, 24, 10, 24> ranlux24_base;

// XXX N2111 uses uint_fast64_t here
typedef subtract_with_carry_engine<thrust::detail::uint64_t, 48,  5, 12> ranlux48_base;


} // end random

// import names into thrust::
using random::subtract_with_carry_engine;
using random::ranlux24_base;
using random::ranlux48_base;

} // end thrust

#include <thrust/random/detail/subtract_with_carry_engine.inl>

