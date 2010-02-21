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


/*! \file uniform_real_distribution.h
 *  \brief A uniform distribution of integer-valued numbers.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/pair.h>
#include <thrust/detail/integer_traits.h>
#include <thrust/random/detail/random_core_access.h>
#include <iostream>

namespace thrust
{

namespace random
{


template<typename IntType = int>
  class uniform_int_distribution
{
  public:
    // types
    typedef IntType result_type;
    typedef thrust::pair<int,int> param_type;

    // constructors and reset functions
    __host__ __device__
    explicit uniform_int_distribution(IntType a = 0, IntType b = thrust::detail::integer_traits<IntType>::const_max);

    __host__ __device__
    explicit uniform_int_distribution(const param_type &parm);

    __host__ __device__
    void reset(void);

    // generating functions
    template<typename UniformRandomNumberGenerator>
    __host__ __device__
    result_type operator()(UniformRandomNumberGenerator &urng);

    template<typename UniformRandomNumberGenerator>
    __host__ __device__
    result_type operator()(UniformRandomNumberGenerator &urng, const param_type &parm);

    // property functions
    __host__ __device__
    result_type a(void) const;

    __host__ __device__
    result_type b(void) const;

    __host__ __device__
    param_type param(void) const;

    __host__ __device__
    void param(const param_type &parm);

    __host__ __device__
    result_type min(void) const;

    __host__ __device__
    result_type max(void) const;

    /*! \cond
     */
  private:
    param_type m_param;

    friend struct thrust::random::detail::random_core_access;

    __host__ __device__
    bool equal(const uniform_int_distribution &rhs) const;

    template<typename CharT, typename Traits>
    std::basic_ostream<CharT,Traits>& stream_out(std::basic_ostream<CharT,Traits> &os) const;

    template<typename CharT, typename Traits>
    std::basic_istream<CharT,Traits>& stream_in(std::basic_istream<CharT,Traits> &is);
    /*! \endcond
     */
}; // end uniform_int_distribution


template<typename IntType>
__host__ __device__
bool operator==(const uniform_int_distribution<IntType> &lhs,
                const uniform_int_distribution<IntType> &rhs);

template<typename IntType>
__host__ __device__
bool operator!=(const uniform_int_distribution<IntType> &lhs,
                const uniform_int_distribution<IntType> &rhs);

template<typename IntType,
         typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os,
           const uniform_int_distribution<IntType> &d);

template<typename IntType,
         typename CharT, typename Traits>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits> &is,
           uniform_int_distribution<IntType> &d);


} // end random

using random::uniform_int_distribution;

} // end thrust

#include <thrust/random/detail/uniform_int_distribution.inl>

