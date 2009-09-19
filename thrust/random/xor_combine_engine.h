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

namespace thrust
{

namespace experimental
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
    // XXX result_type is larger of
    // base1_type::result_type & base2_type::result_type
    typedef ... result_type;
    
    static const size_t shift1 = s1;
    static const size_t shift2 = s2;
    static const result_type min = 0;
    static const result_type max = ...;

    __host__ __device__
    xor_combine_engine(void);

    __host__ __device__
    xor_combine_engine(const base1_type &rng1, const base2_type &rng2);

    __host__ __device__
    xor_combine_engine(unsigned long s);

    template<typename Gen>
    __host__ __device__
    xor_combine_engine(Gen &g);

    __host__ __device__
    void seed(void);

    template<typename Gen>
    __host__ __device__
    void seed(Gen &g);

    __host__ __device__
    const base_type1 &base1(void) const;

    __host__ __device__
    const base_type2 &base2(void) const;

    __host__ __device__
    result_type operator()(void);

    /*! \cond
     */
  private:
    base_type1 m_b1;
    base_type2 m_b2;
    /*! \endcond
     */
}; // end xor_combine_engine

} // end random

} // end experimental

} // end thrust

#include <thrust/random/detail/xor_combine_engine.inl>

