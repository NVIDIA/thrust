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


/*! \file uniform_real.h
 *  \brief A uniform distribution of real-valued numbers.
 */

#pragma once

namespace thrust
{

namespace experimental
{

namespace random
{

template<typename RealType = double>
  class uniform_real
{
  public:
    typedef RealType input_type;
    typedef RealType result_type;

    __host__ __device__
    explicit uniform_real(RealType min = RealType(0), RealType max = RealType(1));

    __host__ __device__
    result_type min(void) const;

    __host__ __device__
    result_type max(void) const;

    __host__ __device__
    void reset(void) const;

    template<typename UniformRandomNumberGenerator>
    __host__ __device__
      result_type operator()(UniformRandomNumberGenerator &urng);

  protected:
    RealType m_min, m_max;
}; // end uniform_real

} // end random

} // end experimental

} // end thrust

#include <thrust/random/detail/uniform_real.inl>

