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


/*! \file device_vector.inl
 *  \brief Inline file for device_vector.h.
 */

#include <thrust/host_vector.h>

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

} // end namespace thrust

