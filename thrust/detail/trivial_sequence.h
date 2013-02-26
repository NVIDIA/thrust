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

/*! \file trivial_sequence.h
 *  \brief Container-like class for wrapping sequences.  The wrapped
 *         sequence always has trivial iterators, even when the input
 *         sequence does not.
 */


#pragma once

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/detail/temporary_array.h>

namespace thrust
{

namespace detail
{

// never instantiated
template<typename Iterator, typename DerivedPolicy, typename is_trivial> struct _trivial_sequence { };

// trivial case
template<typename Iterator, typename DerivedPolicy>
struct _trivial_sequence<Iterator, DerivedPolicy, thrust::detail::true_type>
{
    typedef Iterator iterator_type;
    Iterator first, last;

    _trivial_sequence(thrust::execution_policy<DerivedPolicy> &, Iterator _first, Iterator _last) : first(_first), last(_last)
    {
//        std::cout << "trivial case" << std::endl;
    }

    iterator_type begin() { return first; }
    iterator_type end()   { return last; }
};

// non-trivial case
template<typename Iterator, typename DerivedPolicy>
struct _trivial_sequence<Iterator, DerivedPolicy, thrust::detail::false_type>
{
    typedef typename thrust::iterator_value<Iterator>::type iterator_value;
    typedef typename thrust::detail::temporary_array<iterator_value, DerivedPolicy>::iterator iterator_type;
    
    thrust::detail::temporary_array<iterator_value, DerivedPolicy> buffer;

    _trivial_sequence(thrust::execution_policy<DerivedPolicy> &exec, Iterator first, Iterator last)
      : buffer(exec, first, last)
    {
//        std::cout << "non-trivial case" << std::endl;
    }

    iterator_type begin() { return buffer.begin(); }
    iterator_type end()   { return buffer.end(); }
};

template <typename Iterator, typename DerivedPolicy>
struct trivial_sequence
  : detail::_trivial_sequence<Iterator, DerivedPolicy, typename thrust::detail::is_trivial_iterator<Iterator>::type>
{
    typedef _trivial_sequence<Iterator, DerivedPolicy, typename thrust::detail::is_trivial_iterator<Iterator>::type> super_t;

    trivial_sequence(thrust::execution_policy<DerivedPolicy> &exec, Iterator first, Iterator last) : super_t(exec, first, last) { }
};

} // end namespace detail

} // end namespace thrust

