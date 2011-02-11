/*
 *  Copyright 2008-2011 NVIDIA Corporation
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

/*! \file placeholders.h
 *  \brief Placeholders for lambda expressions.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/lambda/detail/actor.h>
#include <thrust/lambda/detail/argument.h>

namespace thrust
{

namespace lambda
{

namespace placeholders
{

extern const thrust::lambda::detail::actor<thrust::lambda::detail::argument<0> >  _1;
extern const thrust::lambda::detail::actor<thrust::lambda::detail::argument<1> >  _2;
extern const thrust::lambda::detail::actor<thrust::lambda::detail::argument<2> >  _3;
extern const thrust::lambda::detail::actor<thrust::lambda::detail::argument<3> >  _4;
extern const thrust::lambda::detail::actor<thrust::lambda::detail::argument<4> >  _5;
extern const thrust::lambda::detail::actor<thrust::lambda::detail::argument<5> >  _6;
extern const thrust::lambda::detail::actor<thrust::lambda::detail::argument<6> >  _7;
extern const thrust::lambda::detail::actor<thrust::lambda::detail::argument<7> >  _8;
extern const thrust::lambda::detail::actor<thrust::lambda::detail::argument<8> >  _9;
extern const thrust::lambda::detail::actor<thrust::lambda::detail::argument<9> >  _10;

} // end placeholders

} // end lambda

namespace placeholders = thrust::lambda::placeholders;

} // end thrust

