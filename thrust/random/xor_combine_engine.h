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

/*! \file xor_combine_engine.h
 *  \brief A pseudorandom number generator which produces pseudorandom
 *         numbers from two integer base engines by merging their
 *         pseudorandom numbers with bitwise exclusive-or.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/random/detail/xor_combine_engine_max.h>
#include <iostream>

namespace thrust
{

namespace random
{

template<typename Engine1, size_t s1,
         typename Engine2, size_t s2=0u>
  class xor_combine_engine
{
  public:
    typedef Engine1 base1_type;
    typedef Engine2 base2_type;

    typedef typename thrust::detail::eval_if<
      (sizeof(typename base2_type::result_type) > sizeof(typename base1_type::result_type)),
      thrust::detail::identity_<typename base2_type::result_type>,
      thrust::detail::identity_<typename base1_type::result_type>
    >::type result_type;
    
    static const size_t shift1 = s1;
    static const size_t shift2 = s2;
    static const result_type min = 0;
    static const result_type max =
      detail::xor_combine_engine_max<
        Engine1, s1, Engine2, s2, result_type
      >::value;

    __host__ __device__
    xor_combine_engine(void);

    __host__ __device__
    xor_combine_engine(const base1_type &urng1, const base2_type &urng2);

    __host__ __device__
    xor_combine_engine(result_type s);

    __host__ __device__
    void seed(void);

    __host__ __device__
    void seed(result_type s);

    __host__ __device__
    const base1_type &base1(void) const;

    __host__ __device__
    const base2_type &base2(void) const;

    __host__ __device__
    result_type operator()(void);

    __host__ __device__
    void discard(unsigned long long z);

    template<typename Engine1_, size_t s1_, typename Engine2_, size_t s2_>
    friend __host__ __device__
    bool operator==(const xor_combine_engine<Engine1_,s1_,Engine2_,s2_> &lhs,
                    const xor_combine_engine<Engine1_,s1_,Engine2_,s2_> &rhs);

    template<typename Engine1_, size_t s1_, typename Engine2_, size_t s2_>
    friend __host__ __device__
    bool operator!=(const xor_combine_engine<Engine1_,s1_,Engine2_,s2_> &lhs,
                    const xor_combine_engine<Engine1_,s1_,Engine2_,s2_> &rhs);

    template<typename Engine1_, size_t s1_, typename Engine2_, size_t s2_,
             typename CharT, typename Traits>
    friend std::basic_ostream<CharT,Traits>&
    operator<<(std::basic_ostream<CharT,Traits> &os,
               const xor_combine_engine<Engine1_,s1_,Engine2_,s2_> &e);

    template<typename Engine1_, size_t s1_, typename Engine2_, size_t s2_,
             typename CharT, typename Traits>
    friend std::basic_istream<CharT,Traits>&
    operator>>(std::basic_istream<CharT,Traits> &is,
               xor_combine_engine<Engine1_,s1_,Engine2_,s2_> &e);

    /*! \cond
     */
  private:
    base1_type m_b1;
    base2_type m_b2;
    /*! \endcond
     */
}; // end xor_combine_engine

} // end random

} // end thrust

#include <thrust/random/detail/xor_combine_engine.inl>

