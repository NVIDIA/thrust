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


/*! \file uniform_real_distribution.h
 *  \brief A uniform distribution of real-valued numbers.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/pair.h>

namespace thrust
{

namespace random
{

namespace experimental
{

template<typename RealType = double>
  class uniform_real_distribution
{
  public:
    // types
    typedef RealType result_type;
    typedef thrust::pair<RealType,RealType> param_type;

    // constructors and reset functions
    __host__ __device__
    explicit uniform_real_distribution(RealType a = 0.0, RealType b = 1.0);

    __host__ __device__
    explicit uniform_real_distribution(const param_type &parm);

    __host__ __device__
    uniform_real_distribution(const uniform_real_distribution &x);

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

  protected:
    RealType m_a, m_b;
}; // end uniform_real_distribution

} // end experimental

} // end random

} // end thrust

#include <thrust/random/detail/uniform_real_distribution.inl>

