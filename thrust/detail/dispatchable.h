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


#pragma once

#include <thrust/detail/config.h>

namespace thrust
{


template<typename Derived>
  struct dispatchable
{
  // XXX eliminate this later
  typedef Derived derived_type;

  __host__ __device__
  inline derived_type &derived()
  {
    return static_cast<derived_type&>(*this);
  }

  __host__ __device__
  inline const derived_type &derived() const
  {
    return static_cast<const derived_type&>(*this);
  }
};


} // end thrust

