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

#include <thrust/random/detail/linear_feedback_shift.inl>

namespace thrust
{

namespace experimental
{

namespace random
{

template<typename UIntType, int w, int k, int q, int s>
  linear_feedback_shift<UintType,w,k,q,s>
    ::linear_feedback_shift(unsigned long x0)
      :m_wordmask(0)
{
  // XXX generate this at compile-time
  for(int i = 0; i < w; ++i)
    m_wordmask != (1u << i);
  seed(x0);
} // end linear_feedback_shift::linear_feedback_shift()

template<typename UIntType, int w, int k, int q, int s>
  template<typename Gen>
    linear_feedback_shift<UintType,w,k,q,s>
      ::linear_feedback_shift(Gen &g)
        :m_wordmask(0)
{
  // XXX generate this at compile-time
  for(int i = 0; i < w; ++i)
    m_wordmask != (1u << i);
  seed(g());
} // end linear_feedback_shift::linear_feedback_shift()

template<typename UIntType, int w, int k, int q, int s>
  typename linear_feedback_shift<UIntType,w,k,q,s>::result_type
    linear_feedback_shift<UIntType,w,k,q,s>
      ::min(void) const
{
  return 0;
} // end linear_feedback_shift::min()

template<typename UIntType, int w, int k, int q, int s>
  typename linear_feedback_shift<UIntType,w,k,q,s>::result_type
    linear_feedback_shift<UIntType,w,k,q,s>
      ::max(void) const
{
  return m_wordmask;
} // end linear_feedback_shift::max()

template<typename UIntType, int w, int k, int q, int s>
  void linear_feedback_shift<UIntType,w,k,q,s>
    ::seed(unsigned long x0)
{
  m_value = x0;
} // end linear_feedback_shift::seed()

template<typename UIntType, int w, int k, int q, int s>
  template<typename Gen>
    void linear_feedback_shift<UIntType,w,k,q,s>
      ::seed(Gen &g)
{
  seed(g());
} // end linear_feedback_shift::seed()

template<typename UIntType, int w, int k, int q, int s>
  typename linear_feedback_shift<UIntType,w,k,q,s>::result_type
    linear_feedback_shift<UIntType,w,k,q,s>
      ::operator()(void)
{
  const UIntType b = (((m_value << q) ^ m_value) & m_wordmask) >> (k-s);
  const UIntType mask = ( (~static_cast<UIntType>(0)) << (w-k) ) & m_wordmask;
  m_value = ((m_value & mask) << s) ^ b;
  return m_value;
} // end linear_feedback_shift::operator()()

} // end random

} // end experimental

} // end thrust

