/*
 *  Copyright 2008-2010 NVIDIA Corporation
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

//  Copyright Thorsten Ottosen 2003-2004. Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
// For more information, see http://www.boost.org/libs/range/

#pragma once


#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/range/detail/const_iterator.h>
#include <thrust/range/detail/mutable_iterator.h>


namespace thrust
{

namespace experimental
{


template<typename Range>
  struct range_iterator
    : thrust::detail::eval_if<
        thrust::detail::is_const<Range>::value,
        range_const_iterator<
          typename thrust::detail::remove_const<Range>::type
        >,
        range_mutable_iterator<Range>
      >
{};


} // end experimental

} // end thrust

