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

#include <thrust/random/uniform_real_distribution.h>

namespace thrust
{

namespace experimental
{

namespace random
{

template<typename RealType>
  uniform_real_distribution<RealType>
    ::uniform_real_distribution(RealType a, RealType b)
      :m_a(a),m_b(b)
{
} // end uniform_real_distribution::uniform_real_distribution()

template<typename RealType>
  uniform_real_distribution<RealType>
    ::uniform_real_distribution(const param_type &parm)
      :m_a(parm.first),m_b(parm.second)
{
} // end uniform_real_distribution::uniform_real_distribution()

template<typename RealType>
  uniform_real_distribution<RealType>
    ::uniform_real_distribution(const uniform_real_distribution &x)
      :m_a(x.m_a),m_b(x.m_b)
{
} // end uniform_real_distribution::uniform_real_distribution()

template<typename RealType>
  void uniform_real_distribution<RealType>
    ::reset(void)
{
} // end uniform_real_distribution::reset()

template<typename RealType>
  template<typename UniformRandomNumberGenerator>
    typename uniform_real_distribution<RealType>::result_type
      uniform_real_distribution<RealType>
        ::operator()(UniformRandomNumberGenerator &urng)
{
  return operator()(urng, thrust::make_pair(a(), b()));
} // end uniform_real::operator()()

template<typename RealType>
  template<typename UniformRandomNumberGenerator>
    typename uniform_real_distribution<RealType>::result_type
      uniform_real_distribution<RealType>
        ::operator()(UniformRandomNumberGenerator &urng,
                     const param_type &parm)
{
  // call the urng & map its result to [0,1]
  RealType result = RealType(urng() - UniformRandomNumberGenerator::min);
  result /= (UniformRandomNumberGenerator::max - UniformRandomNumberGenerator::min);

  return (result * (parm.second - parm.first)) + parm.first;
} // end uniform_real::operator()()

template<typename RealType>
  typename uniform_real_distribution<RealType>::result_type
    uniform_real_distribution<RealType>
      ::a(void) const
{
  return m_a;
} // end uniform_real::a()

template<typename RealType>
  typename uniform_real_distribution<RealType>::result_type
    uniform_real_distribution<RealType>
      ::b(void) const
{
  return m_b;
} // end uniform_real_distribution::b()

template<typename RealType>
  typename uniform_real_distribution<RealType>::param_type
    uniform_real_distribution<RealType>
      ::param(void) const
{
  return thrust::make_pair(a(),b());
} // end uniform_real_distribution::param()

template<typename RealType>
  void uniform_real_distribution<RealType>
    ::param(const param_type &parm)
{
  m_a = parm.first;
  m_b = parm.second;
} // end uniform_real_distribution::param()

template<typename RealType>
  typename uniform_real_distribution<RealType>::result_type
    uniform_real_distribution<RealType>
      ::min(void) const
{
  return a();
} // end uniform_real_distribution::min()

template<typename RealType>
  typename uniform_real_distribution<RealType>::result_type
    uniform_real_distribution<RealType>
      ::max(void) const
{
  return b();
} // end uniform_real_distribution::max()


} // end random

} // end experimental

} // end thrust

