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
#include <thrust/detail/type_traits.h>

/*
 * (C) Copyright John Maddock 2000.
 * 
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
 */

//
// Helper macro THRUST_JOIN (based on BOOST_JOIN):
// The following piece of macro magic joins the two
// arguments together, even when one of the arguments is
// itself a macro (see 16.3.1 in C++ standard).  The key
// is that macro expansion of macro arguments does not
// occur in THRUST_DO_JOIN2 but does in THRUST_DO_JOIN.
//
#define THRUST_JOIN( X, Y ) THRUST_DO_JOIN( X, Y )
#define THRUST_DO_JOIN( X, Y ) THRUST_DO_JOIN2(X,Y)
#define THRUST_DO_JOIN2( X, Y ) X##Y

namespace thrust
{

namespace detail
{

// HP aCC cannot deal with missing names for template value parameters
template <bool x> struct STATIC_ASSERTION_FAILURE;

template <> struct STATIC_ASSERTION_FAILURE<true> { enum { value = 1 }; };

// HP aCC cannot deal with missing names for template value parameters
template<int x> struct static_assert_test{};

template<typename, bool x>
  struct depend_on_instantiation
{
  static const bool value = x;
};

} // end detail

} // end thrust

#define THRUST_STATIC_ASSERT( B ) \
   typedef ::thrust::detail::static_assert_test<\
      sizeof(::thrust::detail::STATIC_ASSERTION_FAILURE< (bool)( B ) >)>\
         THRUST_JOIN(thrust_static_assert_typedef_, __LINE__)

