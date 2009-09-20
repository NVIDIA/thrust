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

#include <thrust/random/linear_congruential.h>
#include <thrust/random/detail/linear_congruential.inl>
#include <thrust/random/detail/const_mod.h>

namespace thrust
{

namespace experimental
{

namespace random
{


template<typename UIntType, UIntType a, UIntType c, UIntType m>
  linear_congruential<UIntType,a,c,m>
    ::linear_congruential(result_type s)
{
  seed(s);
} // end linear_congruential::linear_congruential()


template<typename UIntType, UIntType a, UIntType c, UIntType m>
  void linear_congruential
    ::seed(unsigned long x0 = 1)
{
  if(increment % modulus == 0 && x0 % modulus == 0)
    m_x = 1 % modulus;
  else
    m_x = x0 % modulus;
} // end linear_congruential::seed()


template<typename UIntType, UIntType a, UIntType c, UIntType m>
  template<typename Gen>
    void linear_congruential<UIntType,a,c,m>
      ::seed(Gen &g)
{
  seed(g());
} // end linear_congruential::seed()


template<typename UIntType, UIntType a, UIntType c, UIntType m>
  typename linear_congruential<UIntType,a,c,m>::result_type
    linear_congruential<UIntType,a,c,m>
      ::operator()(void)
{
  m_x = (multiplier * m_x + increment) % modulus;
  return m_x;
} // end linear_congruential::operator()()


template<typename UIntType, UIntType a, UIntType c, UIntType m>
  void linear_congruential<UIntType,a,c,m>
    ::discard(unsigned long long z)
{
  for(; z > 0; --z)
  {
    this->operator()();
  }
} // end linear_congruential::discard()

} // end random

} // end experimental

} // end thrust

