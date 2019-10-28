/*
 *  Copyright 2018 NVIDIA Corporation
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

#if   __cplusplus < 201103L
  #define THRUST_CPP03
  #define THRUST_CPP_DIALECT 2003
#elif __cplusplus < 201402L
  #define THRUST_CPP11
  #define THRUST_CPP_DIALECT 2011
#elif __cplusplus < 201703L
  #define THRUST_CPP14
  #define THRUST_CPP_DIALECT 2014
#else
  #define THRUST_CPP17
  #define THRUST_CPP_DIALECT 2017
#endif

