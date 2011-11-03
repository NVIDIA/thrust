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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/tagged_iterator.h>

namespace thrust
{


template<typename Tag, typename Iterator>
  thrust::detail::tagged_iterator<Iterator,Tag>
    reinterpret_tag(Iterator iter)
{
  return thrust::detail::tagged_iterator<Iterator,Tag>(iter);
} // end reinterpret_tag()


// avoid deeply-nested tagged_iterator
template<typename Tag, typename BaseIterator, typename OtherTag>
  thrust::detail::tagged_iterator<BaseIterator,Tag>
    reinterpret_tag(thrust::detail::tagged_iterator<BaseIterator,OtherTag> iter)
{
  return reinterpret_tag<Tag>(iter.base());
} // end reinterpret_tag()


template<typename Tag, typename Iterator>
  typename thrust::detail::enable_if_convertible<
    typename thrust::iterator_space<Iterator>::type,
    Tag,
    thrust::detail::tagged_iterator<Iterator,Tag>
  >::type
    retag(Iterator iter)
{
  return reinterpret_tag<Tag>(iter);
} // end retag()


// avoid deeply-nested tagged_iterator
template<typename Tag, typename BaseIterator, typename OtherTag>
  typename thrust::detail::enable_if_convertible<
    OtherTag,
    Tag,
    thrust::detail::tagged_iterator<BaseIterator,Tag>
  >::type
    retag(thrust::detail::tagged_iterator<BaseIterator,OtherTag> iter)
{
  return reinterpret_tag<Tag>(iter);
} // end retag()


} // end thrust

