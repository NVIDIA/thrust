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


/*! \file discard_block_engine.h
 *  \brief A random number engine which adapts a base engine and produces
 *         numbers by discarding all but a contiguous blocks of its values.
 */

#pragma once

#include <thrust/detail/config.h>

#include <thrust/detail/config.h>
#include <iostream>
#include <thrust/detail/cstdint.h>
#include <thrust/random/detail/random_core_access.h>

namespace thrust
{

namespace random
{

template<typename Engine, size_t p, size_t r>
  class discard_block_engine
{
  public:
    // types
    typedef Engine base_type;
    typedef typename base_type::result_type result_type;

    // engine characteristics
    static const size_t block_size = p;
    static const size_t used_block = r;
    static const result_type min = base_type::min;
    static const result_type max = base_type::max;

    // constructors and seeding functions
    __host__ __device__
    discard_block_engine();

    __host__ __device__
    explicit discard_block_engine(const base_type &urng);

    __host__ __device__
    explicit discard_block_engine(result_type s);

    __host__ __device__
    void seed(void);

    __host__ __device__
    void seed(result_type s);

    // generating functions
    __host__ __device__
    result_type operator()(void);

    __host__ __device__
    void discard(unsigned long long z);

    // property functions
    __host__ __device__
    const base_type &base(void) const;

    /*! \cond
     */
  private:
    base_type m_e;
    int m_n;

    friend struct thrust::random::detail::random_core_access;

    __host__ __device__
    bool equal(const discard_block_engine &rhs) const;

    template<typename CharT, typename Traits>
    std::basic_ostream<CharT,Traits>& stream_out(std::basic_ostream<CharT,Traits> &os) const;

    template<typename CharT, typename Traits>
    std::basic_istream<CharT,Traits>& stream_in(std::basic_istream<CharT,Traits> &is);
    /*! \endcond
     */
}; // end discard_block_engine


template<typename Engine, size_t p, size_t r>
__host__ __device__
bool operator==(const discard_block_engine<Engine,p,r> &lhs,
                const discard_block_engine<Engine,p,r> &rhs);

template<typename Engine, size_t p, size_t r>
__host__ __device__
bool operator!=(const discard_block_engine<Engine,p,r> &lhs,
                const discard_block_engine<Engine,p,r> &rhs);

template<typename Engine, size_t p, size_t r,
         typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os,
           const discard_block_engine<Engine,p,r> &e);

template<typename Engine, size_t p, size_t r,
         typename CharT, typename Traits>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits> &is,
           discard_block_engine<Engine,p,r> &e);

} // end random

// import names into thrust::
using random::discard_block_engine;

} // end thrust

#include <thrust/random/detail/discard_block_engine.inl>

