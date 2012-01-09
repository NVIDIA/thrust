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

#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits/iterator/is_output_iterator.h>

namespace thrust
{
namespace detail
{


// specialize is_output_iterator for discard_iterator: it is a pure output iterator
template<typename Space>
  struct is_output_iterator<thrust::discard_iterator<Space> >
    : thrust::detail::true_type
{
};


} // end namespace detail
} // end namespace thrust

