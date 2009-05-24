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


/*! \file device_vector.inl
 *  \brief Inline file for device_vector.h.
 */

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/equal.h>
#include <thrust/device_ptr.h>

namespace thrust
{

template<typename T, typename Alloc>
  template<typename OtherT, typename OtherAlloc>
    device_vector<T,Alloc>
      ::device_vector(const host_vector<OtherT,OtherAlloc> &v)
        :Parent(v)
{
  ;
} // end device_vector::device_vector()

namespace detail
{

template<typename Vector1,
         typename Vector2>
  bool vectors_equal(const Vector1 &lhs,
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
} // end vectors_equal()

}; // end detail

template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
  bool operator==(const device_vector<T1,Alloc1> &lhs,
                  const device_vector<T2,Alloc2> &rhs)
{
  return thrust::detail::vectors_equal(lhs,rhs);
} // end operator==()

template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
  bool operator==(const host_vector<T1,Alloc1> &lhs,
                  const device_vector<T2,Alloc2> &rhs)
{
  // create temporary host_vector
  host_vector<T2> tempRhs(rhs);

  // compare on host
  return lhs == tempRhs;
} // end operator==()

// allow comparison between device_vector & host_vector
template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
  bool operator==(const device_vector<T1,Alloc1> &lhs,
                  const host_vector<T2,Alloc2> &rhs)
{
  // create temporary host_vector
  host_vector<T1> tempLhs(lhs);

  // compare on host
  return tempLhs == rhs;
} // end operator==()

// allow comparison between std::vector & device_vector
template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
  bool operator==(const std::vector<T1,Alloc1> &lhs,
                  const device_vector<T2,Alloc2> &rhs)
{
  // create temporary host_vector
  host_vector<T2> tempRhs(rhs);

  // compare on host
  return lhs == tempRhs;
} // end operator==()

// allow comparison between device_vector & std::vector
template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
  bool operator==(const device_vector<T1,Alloc1> &lhs,
                  const std::vector<T2,Alloc2> &rhs)
{
  // create temporary host_vector
  host_vector<T1> tempLhs(lhs);

  // compare on host
  return tempLhs == rhs;
} // end operator==()

} // end thrust

