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

/*! \file linear_feedback_shift.h
 *  \brief A linear feedback shift pseudorandom number generator.
 */

#pragma once

namespace thrust
{

namespace experimental
{

namespace random
{

template<typename UIntType, int w, int k, int q, int s>
  class linear_feedback_shift
{
  public:
    typedef UIntType result_type;

    static const int word_size = w;
    static const int exponent1 = k;
    static const int exponent2 = q;
    static const int step_size = s;

    __host__ __device__
    explicit linear_feedback_shift(unsigned long x0 = 341);

    template<typename Gen>
    __host__ __device__
    linear_feedback_shift(Gen &g);

    __host__ __device__
    void seed(unsigned long x0 = 341);

    template<typename Gen>
    __host__ __device__
    void seed(Gen &g);

    __host__ __device__
    result_type min(void) const;

    __host__ __device__
    result_type max(void) const;

    __host__ __device__
    result_type operator()(void);

    /*! \cond
     */
  private:
    UIntType m_wordmask, m_value;
    /*! \endcond
     */
}; // end linear_feedback_shift

} // end random

} // end experimental

} // end thrust

#include <thrust/random/detail/linear_feedback_shift.inl>

