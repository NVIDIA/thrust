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

#include <thrust/random/linear_feedback_shift_engine.h>

namespace thrust
{

namespace experimental
{

namespace random
{

template<typename UIntType, int w, int k, int q, int s>
  linear_feedback_shift_engine<UIntType,w,k,q,s>
    ::linear_feedback_shift_engine(result_type value)
{
  seed(value);
} // end linear_feedback_shift_engine::linear_feedback_shift_engine()

template<typename UIntType, int w, int k, int q, int s>
  void linear_feedback_shift_engine<UIntType,w,k,q,s>
    ::seed(result_type value)
{
  m_value = value;
} // end linear_feedback_shift_engine::seed()

template<typename UIntType, int w, int k, int q, int s>
  typename linear_feedback_shift_engine<UIntType,w,k,q,s>::result_type
    linear_feedback_shift_engine<UIntType,w,k,q,s>
      ::operator()(void)
{
  const UIntType b = (((m_value << q) ^ m_value) & wordmask) >> (k-s);
  const UIntType mask = ( (~static_cast<UIntType>(0)) << (w-k) ) & wordmask;
  m_value = ((m_value & mask) << s) ^ b;
  return m_value;
} // end linear_feedback_shift_engine::operator()()

template<typename UIntType, int w, int k, int q, int s>
  void linear_feedback_shift_engine<UIntType,w,k,q,s>
    ::discard(unsigned long long z)
{
  for(; z > 0; --z)
  {
    this->operator()();
  } // end for
} // end linear_feedback_shift_engine::discard()

} // end random

} // end experimental

} // end thrust

