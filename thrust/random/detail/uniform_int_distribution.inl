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

#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/detail/type_traits.h>
#include <math.h>

namespace thrust
{

namespace random
{

namespace experimental
{

template<typename IntType>
  uniform_int_distribution<IntType>
    ::uniform_int_distribution(IntType a, IntType b)
      :m_param(a,b)
{
} // end uniform_int_distribution::uniform_int_distribution()


template<typename IntType>
  uniform_int_distribution<IntType>
    ::uniform_int_distribution(const param_type &parm)
      :m_param(parm)
{
} // end uniform_int_distribution::uniform_int_distribution()


template<typename IntType>
  void uniform_int_distribution<IntType>
    ::reset(void)
{
} // end uniform_int_distribution::reset()


template<typename IntType>
  template<typename UniformRandomNumberGenerator>
    typename uniform_int_distribution<IntType>::result_type
      uniform_int_distribution<IntType>
        ::operator()(UniformRandomNumberGenerator &urng)
{
  return operator()(urng, m_param);
} // end uniform_int_distribution::operator()()


template<typename IntType>
  template<typename UniformRandomNumberGenerator>
    typename uniform_int_distribution<IntType>::result_type
      uniform_int_distribution<IntType>
        ::operator()(UniformRandomNumberGenerator &urng, const param_type &parm)
{
  // XXX this implementation is somewhat hacky and will skip
  //     values if the range of the RNG is smaller than the range of the distribution
  //     we should improve this implementation in a later version

  typedef typename thrust::detail::largest_available_float::type float_type;

  const float_type real_min(parm.first);
  const float_type real_max(parm.second);

  uniform_real_distribution<float_type> real_dist(real_min, detail::nextafter(real_max, real_max + float_type(1)));

  return static_cast<result_type>(real_dist(urng) + float_type(0.5));
} // end uniform_int_distribution::operator()()


template<typename IntType>
  typename uniform_int_distribution<IntType>::result_type
    uniform_int_distribution<IntType>
      ::a(void) const
{
  return m_param.first;
} // end uniform_int_distribution<IntType>::a()


template<typename IntType>
  typename uniform_int_distribution<IntType>::result_type
    uniform_int_distribution<IntType>
      ::b(void) const
{
  return m_param.second;
} // end uniform_int_distribution::b()


template<typename IntType>
  typename uniform_int_distribution<IntType>::param_type
    uniform_int_distribution<IntType>
      ::param(void) const
{
  return m_param;
} // end uniform_int_distribution::param()


template<typename IntType>
  void uniform_int_distribution<IntType>
    ::param(const param_type &parm)
{
  m_param = parm;
} // end uniform_int_distribution::param()


template<typename IntType>
  typename uniform_int_distribution<IntType>::result_type
    uniform_int_distribution<IntType>
      ::min(void) const
{
  return a();
} // end uniform_int_distribution::min()


template<typename IntType>
  typename uniform_int_distribution<IntType>::result_type
    uniform_int_distribution<IntType>
      ::max(void) const
{
  return b();
} // end uniform_int_distribution::max()


} // end experimental

} // end random

} // end thrust

