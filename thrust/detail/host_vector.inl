/*
 *  Copyright 2008-2012 NVIDIA Corporation
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

} // end namespace thrust

