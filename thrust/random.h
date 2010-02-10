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

/*! \file random.h
 *  \brief Pseudo-random number generators.
 */

#pragma once

#include <thrust/detail/config.h>

// RNGs
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/linear_feedback_shift_engine.h>
#include <thrust/random/subtract_with_carry_engine.h>
#include <thrust/random/xor_combine_engine.h>

// distributions
#include <thrust/random/uniform_real_distribution.h>

namespace thrust
{

namespace random
{

typedef xor_combine_engine<
  linear_feedback_shift_engine<uint32_t, 32u, 31u, 13u, 12u>,
  0,
  xor_combine_engine<
    linear_feedback_shift_engine<uint32_t, 32u, 29u,  2u,  4u>, 0,
    linear_feedback_shift_engine<uint32_t, 32u, 28u,  3u, 17u>, 0
  >,
  0
> taus88;

} // end random

} // end thrust

