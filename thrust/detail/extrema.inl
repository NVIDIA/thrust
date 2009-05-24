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

#include <thrust/extrema.h>

namespace thrust
{

template<typename T, typename BinaryPredicate>
  T min(const T &lhs, const T &rhs, BinaryPredicate comp)
{
  return comp(lhs, rhs) ? lhs : rhs;
} // end min()

template<typename T>
  T min(const T &lhs, const T &rhs)
{
  return lhs < rhs ? lhs : rhs;
} // end min()

template<typename T, typename BinaryPredicate>
  T max(const T &lhs, const T &rhs, BinaryPredicate comp)
{
  return comp(lhs,rhs) ? rhs : lhs;
} // end max()

template<typename T>
  T max(const T &lhs, const T &rhs)
{
  return lhs > rhs ? lhs : rhs;
} // end max()

} // end thrust

