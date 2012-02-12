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


/*! \file find.inl
 *  \brief Inline file for find.h
 */

#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/find.h>
#include <thrust/detail/adl_helper.h>

namespace thrust
{

template <typename InputIterator, typename T>
InputIterator find(InputIterator first,
                   InputIterator last,
                   const T& value)
{
    using thrust::system::detail::generic::select_system;
    using thrust::system::detail::generic::find;

    typedef typename thrust::iterator_system<InputIterator>::type system;

    return find(select_system(system()), first, last, value);
}

template <typename InputIterator, typename Predicate>
InputIterator find_if(InputIterator first,
                      InputIterator last,
                      Predicate pred)
{
    using thrust::system::detail::generic::select_system;
    using thrust::system::detail::generic::find_if;

    typedef typename thrust::iterator_system<InputIterator>::type system;

    return find_if(select_system(system()), first, last, pred);
}

template <typename InputIterator, typename Predicate>
InputIterator find_if_not(InputIterator first,
                          InputIterator last,
                          Predicate pred)
{
    using thrust::system::detail::generic::select_system;
    using thrust::system::detail::generic::find_if_not;

    typedef typename thrust::iterator_system<InputIterator>::type system;

    return find_if_not(select_system(system()), first, last, pred);
}

} // end namespace thrust

