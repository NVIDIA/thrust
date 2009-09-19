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

#include <thrust/random/xor_combine_engine.h>

namespace thrust
{

namespace experimental
{

namespace random
{

template<typename Engine1, size_t s1,
         typename Engine2, size_t s2>
  xor_combine_engine<Engine1,s1,Engine2,s2>
    ::xor_combine_engine(void)
      :m_b1(),m_b2()
{
} // end xor_combine_engine::xor_combine_engine()

template<typename Engine1, size_t s1,
         typename Engine2, size_t s2>
  xor_combine_engine<Engine1,s1,Engine2,s2>
    ::xor_combine_engine(const base_type1 &rng1, const base_type2 &rng2)
      :m_b1(rng1),m_b2(rng2)
{
} // end xor_combine_engine::xor_combine_engine()

template<typename Engine1, size_t s1,
         typename Engine2, size_t s2>
  xor_combine_engine<Engine1,s1,Engine2,s2>
    ::xor_combine_engine(unsigned long s)
      :m_b1(s),m_b2(s+1)
{
} // end xor_combine_engine::xor_combine_engine()

template<typename Engine1, size_t s1,
         typename Engine2, size_t s2>
  template<typename Gen>
    xor_combine_engine<Engine1,s1,Engine2,s2>
      ::xor_combine_engine(Gen &g)
        :m_b1(g),m_b2(g)
{
} // end xor_combine_engine::xor_combine_engine()

template<typename Engine1, size_t s1,
         typename Engine2, size_t s2>
  void xor_combine_engine<Engine1,s1,Engine2,s2>
    ::seed(void)
{
  m_b1.seed();
  m_b2.seed();
} // end xor_combine_engine::seed()

template<typename Engine1, size_t s1,
         typename Engine2, size_t s2>
  template<typename Gen>
    void xor_combine_engine<Engine1,s1,Engine2,s2>
      ::seed(Gen &g)
{
  m_b1.seed(g);
  m_b2.seed(g);
} // end xor_combine_engine::seed()

template<typename Engine1, size_t s1,
         typename Engine2, size_t s2>
  const typename xor_combine_engine<Engine1,s1,Engine2,s2>::base_type1 &
    xor_combine_engine<Engine1,s1,Engine2,s2>
      ::base1(void) const
{
  return m_b1;
} // end xor_combine_engine::base1()

template<typename Engine1, size_t s1,
         typename Engine2, size_t s2>
  const typename xor_combine_engine<Engine1,s1,Engine2,s2>::base_type2 &
    xor_combine_engine<Engine1,s1,Engine2,s2>
      ::base2(void) const
{
  return m_b2;
} // end xor_combine_engine::base2()

template<typename Engine1, size_t s1,
         typename Engine2, size_t s2>
  typename xor_combine_engine<Engine1,s1,Engine2,s2>::result_type
    xor_combine_engine<Engine1,s1,Engine2,s2>
      ::operator()(void)
{
  return (result_type(m_b1() - m_b1.min()) << shift1) ^
         (result_type(m_b2() - m_b2.min()) << shift2);
} // end xor_combine_engine::operator()()

} // end random

} // end experimental

} // end thrust

