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

namespace thrust
{

namespace random
{

namespace experimental
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
    /*! \endcond
     */
}; // end uniform_int_distribution

} // end experimental

} // end random

// XXX import random::uniform_int_distribution when it is non-experimental

} // end thrust

#include <thrust/random/detail/uniform_int_distribution.inl>

