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
#include <iostream>
#include <thrust/detail/cstdint.h>
#include <thrust/random/detail/random_core_access.h>

namespace thrust
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
    explicit linear_congruential_engine(result_type s = default_seed);

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

    friend struct thrust::random::detail::random_core_access;

    __host__ __device__
    bool equal(const linear_congruential_engine &rhs) const;

    template<typename CharT, typename Traits>
    std::basic_ostream<CharT,Traits>& stream_out(std::basic_ostream<CharT,Traits> &os) const;

    template<typename CharT, typename Traits>
    std::basic_istream<CharT,Traits>& stream_in(std::basic_istream<CharT,Traits> &is);

    /*! \endcond
     */
}; // end linear_congruential


template<typename UIntType_, UIntType_ a_, UIntType_ c_, UIntType_ m_>
__host__ __device__
bool operator==(const linear_congruential_engine<UIntType_,a_,c_,m_> &lhs,
                const linear_congruential_engine<UIntType_,a_,c_,m_> &rhs);

template<typename UIntType_, UIntType_ a_, UIntType_ c_, UIntType_ m_>
__host__ __device__
bool operator!=(const linear_congruential_engine<UIntType_,a_,c_,m_> &lhs,
                const linear_congruential_engine<UIntType_,a_,c_,m_> &rhs);

template<typename UIntType_, UIntType_ a_, UIntType_ c_, UIntType_ m_,
         typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os,
           const linear_congruential_engine<UIntType_,a_,c_,m_> &e);

template<typename UIntType_, UIntType_ a_, UIntType_ c_, UIntType_ m_,
         typename CharT, typename Traits>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits> &is,
           linear_congruential_engine<UIntType_,a_,c_,m_> &e);


// XXX the type N2111 used here was uint_fast32_t
typedef linear_congruential_engine<thrust::detail::uint32_t, 16807, 0, 2147483647> minstd_rand0;
typedef linear_congruential_engine<thrust::detail::uint32_t, 48271, 0, 2147483647> minstd_rand;
  
} // end random

// import names into thrust::
using random::linear_congruential_engine;
using random::minstd_rand;
using random::minstd_rand0;

} // end thrust

#include <thrust/random/detail/linear_congruential_engine.inl>

