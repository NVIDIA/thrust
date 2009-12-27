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

#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/detail/mod.h>

namespace thrust
{

namespace experimental
{

namespace random
{


template<typename UIntType, UIntType a, UIntType c, UIntType m>
  linear_congruential_engine<UIntType,a,c,m>
    ::linear_congruential_engine(result_type s)
{
  seed(s);
} // end linear_congruential_engine::linear_congruential_engine()


template<typename UIntType, UIntType a, UIntType c, UIntType m>
  void linear_congruential_engine<UIntType,a,c,m>
    ::seed(result_type s)
{
  if((detail::mod<UIntType, 1, 0, m>(c) == 0) &&
     (detail::mod<UIntType, 1, 0, m>(s) == 0))
    m_x = detail::mod<UIntType, 1, 0, m>(1);
  else
    m_x = detail::mod<UIntType, 1, 0, m>(s);
} // end linear_congruential_engine::seed()


template<typename UIntType, UIntType a, UIntType c, UIntType m>
  typename linear_congruential_engine<UIntType,a,c,m>::result_type
    linear_congruential_engine<UIntType,a,c,m>
      ::operator()(void)
{
  m_x = detail::mod<UIntType,a,c,m>(m_x);
  return m_x;
} // end linear_congruential_engine::operator()()


template<typename UIntType, UIntType a, UIntType c, UIntType m>
  void linear_congruential_engine<UIntType,a,c,m>
    ::discard(unsigned long long z)
{
  for(; z > 0; --z)
  {
    this->operator()();
  } // end for
} // end linear_congruential_engine::discard()


} // end random

} // end experimental

} // end thrust

