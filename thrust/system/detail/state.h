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
namespace system
{
namespace detail
{


template<typename Derived>
  struct state
{
  inline Derived &derived()
  {
    return static_cast<Derived&>(*this);
  }

  inline const Derived &derived() const
  {
    return static_cast<const Derived&>(*this);
  }
};


} // end detail
} // end system
} // end thrust

