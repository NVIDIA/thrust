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

#include <thrust/random/uniform_real.h>

namespace thrust
{

namespace experimental
{

namespace random
{

template<typename RealType>
  uniform_real<RealType>
    ::uniform_real(RealType min, RealType max)
      :m_min(min),m_max(max)
{
} // end uniform_real::uniform_real()

template<typename RealType>
  typename uniform_real<RealType>::result_type
    uniform_real<RealType>
      ::min(void) const
{
  return m_min;
} // end uniform_real::min()

template<typename RealType>
  typename uniform_real<RealType>::result_type
    uniform_real<RealType>
      ::max(void) const
{
  return m_max;
} // end uniform_real::max()

template<typename RealType>
  void uniform_real<RealType>
    ::reset(void)
{
} // end uniform_real::reset()

template<typename RealType>
  template<typename UniformRandomNumberGenerator>
    typename uniform_real<RealType>::result_type
      ::operator()(UniformRandomNumberGenerator &urng)
{
  // XXX consider storing delta to avoid computing it each timUniformRandomNumberGeneratore
  return (urng() * (m_max - m_min)) + m_min;
} // end uniform_real::operator()()

} // end random

} // end experimental

} // end thrust

