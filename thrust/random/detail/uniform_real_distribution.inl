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

#include <thrust/random/uniform_real_distribution.h>
#include <math.h>

namespace thrust
{

namespace random
{

namespace experimental
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

namespace detail
{

__host__ __device__ __inline__
double nextafter(double x, double y)
{
  return ::nextafter(x,y);
}

__host__ __device__ __inline__
float nextafter(float x, float y)
{
  return ::nextafterf(x,y);
}

}

template<typename RealType>
  template<typename UniformRandomNumberGenerator>
    typename uniform_real_distribution<RealType>::result_type
      uniform_real_distribution<RealType>
        ::operator()(UniformRandomNumberGenerator &urng,
                     const param_type &parm)
{
  // call the urng & map its result to [0,1]
  result_type result = static_cast<result_type>(urng() - UniformRandomNumberGenerator::min);
  result /= static_cast<result_type>(UniformRandomNumberGenerator::max - UniformRandomNumberGenerator::min);

  // do not include parm.second in the result
  // get the next value after parm.second in the direction of parm.first
  // we need to do this because the range is half-open at parm.second

  return (result * (detail::nextafter(parm.second,parm.first) - parm.first)) + parm.first;
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

} // end experimental

} // end random

} // end thrust

