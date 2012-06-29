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
#include <thrust/system/cpp/detail/tag.h>
#include <thrust/system/tbb/detail/tag.h>
#include <thrust/iterator/detail/any_system_tag.h>
#include <thrust/detail/type_traits.h>

namespace thrust
{
namespace system
{
// put the canonical tag in the same ns as the backend's entry points
namespace omp
{
namespace detail
{

// forward declaration of state
template<typename> struct state;

// tag's specialization comes first
// note we inherit cpp's functionality
template<>
  struct state<void>
    : thrust::system::cpp::detail::state< state<void> >,
      private virtual thrust::system::detail::final
{};

// tag is just a typedef for state<void>
typedef state<void> tag;

// note we inherit cpp's functionality
template<typename Derived>
  struct state
    : thrust::system::cpp::detail::state<Derived>
{
  // allow conversion to tag
  inline operator tag () const
  {
    return tag();
  }
};


// overloads of select_system

// XXX select_system(tbb, omp) & select_system(omp, tbb) are ambiguous
//     because both convert to cpp without these overloads, which we
//     arbitrarily define in the omp backend

template<typename DerivedSystem>
inline __host__ __device__
  typename state<DerivedSystem>::derived_type
    select_system(state<DerivedSystem> s, thrust::system::tbb::tag)
{
  return s.derived();
} // end select_system()


template<typename DerivedSystem>
inline __host__ __device__
  typename state<DerivedSystem>::derived_type
    select_system(thrust::system::tbb::tag, state<DerivedSystem> s)
{
  return s.derived()();
} // end select_system()


} // end detail

// alias the tag here
using thrust::system::omp::detail::tag;

} // end omp
} // end system

// alias omp's tag at top-level
namespace omp
{

using thrust::system::omp::tag;

} // end omp
} // end thrust

