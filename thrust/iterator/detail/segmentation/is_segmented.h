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

#pragma once

#include <thrust/detail/has_nested_type.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/segmentation/bucket_iterator.h>

namespace thrust
{

namespace detail
{


// XXX this is a handy thing to have around -- move it to type_traits.h
__THRUST_DEFINE_HAS_NESTED_TYPE(metafunction_has_result_type, type);


// if bucket_iterator<Iterator>::type exists, then Iterator is segmented
template<typename Iterator>
  struct is_segmented
    : metafunction_has_result_type<
        bucket_iterator<Iterator>
      >
{};


} // end detail

} // end thrust

