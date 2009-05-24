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


/*! \file host_vector.inl
 *  \brief Inline file for host_vector.h.
 */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/equal.h>

namespace thrust
{

template<typename T, typename Alloc>
  template<typename OtherT, typename OtherAlloc>
    host_vector<T,Alloc>
      ::host_vector(const device_vector<OtherT,OtherAlloc> &v)
        :Parent(v)
{
  ;
} // end host_vector::host_vector()

namespace detail
{

template<typename Vector1,
         typename Vector2>
  bool host_vector_vectors_equal(const Vector1 &lhs,
                                 const Vector2 &rhs)
{
  bool result = false;
  if(lhs.size() == rhs.size())
  {
    if(lhs.size() > 0)
    {
      result = thrust::equal(lhs.begin(), lhs.end(), rhs.begin());
    } // end if
    else
    {
      result = true;
    } // end else
  } // end if

  return result;
} // end host_vector_vectors_equal()

} // end detail

template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
  bool operator==(const host_vector<T1,Alloc1> &lhs,
                  const host_vector<T2,Alloc2> &rhs)
{
  return thrust::detail::host_vector_vectors_equal(lhs,rhs);
} // end operator==()

template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
  bool operator==(const host_vector<T1,Alloc1> &lhs,
                  const std::vector<T2,Alloc2> &rhs)
{
  return thrust::detail::host_vector_vectors_equal(lhs,rhs);
} // end operator==()

template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
  bool operator==(const std::vector<T1,Alloc1> &lhs,
                  const host_vector<T2,Alloc2> &rhs)
{
  return thrust::detail::host_vector_vectors_equal(lhs,rhs);
} // end operator==()

} // end thrust

