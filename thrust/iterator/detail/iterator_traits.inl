/*
 *  Copyright 2008-2009 NVIDIA Corporation
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


/*! \file iterator_traits.inl
 *  \brief Inline file for iterator_traits.h.
 */

#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/detail/iterator_category_to_traversal.h>
#include <thrust/detail/type_traits.h>

namespace thrust
{

namespace experimental
{


template<typename Iterator>
  struct iterator_value
{
  typedef typename thrust::iterator_traits<Iterator>::value_type type;
}; // end iterator_value


template<typename Iterator>
  struct iterator_pointer
{
  typedef typename thrust::iterator_traits<Iterator>::pointer type;
}; // end iterator_pointer


template<typename Iterator>
  struct iterator_reference
{
  typedef typename iterator_traits<Iterator>::reference type;
}; // end iterator_reference


template<typename Iterator>
  struct iterator_difference
{
  typedef typename thrust::iterator_traits<Iterator>::difference_type type;
}; // end iterator_difference


template<typename Iterator>
  struct iterator_space
    : detail::iterator_category_to_space<
        typename thrust::iterator_traits<Iterator>::iterator_category
      >
{
}; // end iterator_space


template <typename Iterator>
  struct iterator_traversal
    : detail::iterator_category_to_traversal<
        typename thrust::iterator_traits<Iterator>::iterator_category
      >
{
}; // end iterator_traversal

namespace detail
{

template <typename T>
  struct is_iterator_traversal
    : thrust::detail::is_convertible<T, incrementable_traversal_tag>
{
}; // end is_iterator_traversal


template<typename T>
  struct is_iterator_space
    : detail::or_<
        detail::is_convertible<T, space::any>,
        detail::or_<
          detail::is_convertible<T, space::host>,
          detail::is_convertible<T, space::device>
        >
      >
{
}; // end is_iterator_space

} // end detail

} // end experimental

namespace detail
{

template <typename Iterator> struct iterator_device_reference {};

template <typename T>
  struct iterator_device_reference<T*>
{
  typedef T& type;
}; // end iterator_device_reference


} // end detail

} // end thrust

