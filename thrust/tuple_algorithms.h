/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp11_required.h>

#if THRUST_CPP_DIALECT >= 2011

#include <thrust/detail/type_deduction.h>
#include <thrust/type_traits/integer_sequence.h>

#include <tuple>

THRUST_BEGIN_NS

template <typename Tuple, std::size_t... Is>
auto tuple_subset(Tuple&& t, index_sequence<Is...>)
THRUST_DECLTYPE_RETURNS(std::make_tuple(std::get<Is>(THRUST_FWD(t))...));

THRUST_END_NS

#endif // THRUST_CPP_DIALECT >= 2011

