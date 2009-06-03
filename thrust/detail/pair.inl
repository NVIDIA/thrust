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

#include <thrust/pair.h>

namespace thrust
{

template <typename T1, typename T2>
  pair<T1,T2>
    ::pair(void)
      :first(),second()
{
  ;
} // end pair::pair()


template <typename T1, typename T2>
  pair<T1,T2>
    ::pair(const T1 &x, const T2 &y)
      :first(x),second(y)
{
  ;
} // end pair::pair()


template <typename T1, typename T2>
  template <typename U1, typename U2>
    pair<T1,T2>
      ::pair(const pair<U1,U2> &p)
        :first(p.first),second(p.second)
{
  ;
} // end pair::pair()


template <typename T1, typename T2>
  template <typename U1, typename U2>
    pair<T1,T2>
      ::pair(const std::pair<U1,U2> &p)
        :first(p.first),second(p.second)
{
  ;
} // end pair::pair()


template <typename T1, typename T2>
  inline __host__ __device__
    bool operator==(const pair<T1,T2> &x, const pair<T1,T2> &y)
{
  return x.first == y.first && x.second == y.second;
} // end operator==()


template <typename T1, typename T2>
  inline __host__ __device__
    bool operator<(const pair<T1,T2> &x, const pair<T1,T2> &y)
{
  return x.first < y.first || (!(y.first < x.first) && x.second < y.second);
} // end operator<()


template <typename T1, typename T2>
  inline __host__ __device__
    bool operator!=(const pair<T1,T2> &x, const pair<T1,T2> &y)
{
  return !(x == y);
} // end operator==()


template <typename T1, typename T2>
  inline __host__ __device__
    bool operator>(const pair<T1,T2> &x, const pair<T1,T2> &y)
{
  return y < x;
} // end operator<()


template <typename T1, typename T2>
  inline __host__ __device__
    bool operator<=(const pair<T1,T2> &x, const pair<T1,T2> &y)
{
  return !(y < x);
} // end operator<=()


template <typename T1, typename T2>
  inline __host__ __device__
    bool operator>=(const pair<T1,T2> &x, const pair<T1,T2> &y)
{
  return !(x < y);
} // end operator>=()


template <typename T1, typename T2>
  inline __host__ __device__
    pair<T1,T2> make_pair(T1 x, T2 y)
{
  return pair<T1,T2>(x,y);
} // end make_pair()


} // end thrust

