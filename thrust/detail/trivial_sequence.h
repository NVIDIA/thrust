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

/*! \file trivial_sequence.h
 *  \brief Container-like class for wrapping sequences.  The wrapped
 *         sequence always has trivial iterators, even when the input
 *         sequence does not.
 */


#pragma once

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/raw_buffer.h>

namespace thrust
{

namespace detail
{

// never instantiated
template<typename Iterator, typename is_trivial> struct _trivial_sequence { };

// trivial case
template<typename Iterator>
struct _trivial_sequence<Iterator, thrust::detail::true_type>
{
    typedef Iterator iterator_type;
    Iterator first, last;

    _trivial_sequence(Iterator _first, Iterator _last) : first(_first), last(_last)
    {
//        std::cout << "trivial case" << std::endl;
    }

    iterator_type begin() { return first; }
    iterator_type end()   { return last; }
};

// non-trivial case
template<typename Iterator>
struct _trivial_sequence<Iterator, thrust::detail::false_type>
{
    typedef typename thrust::iterator_space<Iterator>::type iterator_space;
    typedef typename thrust::iterator_value<Iterator>::type iterator_value;
    typedef typename thrust::detail::raw_buffer<iterator_value, iterator_space>::iterator iterator_type;
    
    thrust::detail::raw_buffer<iterator_value, iterator_space> buffer;

    _trivial_sequence(Iterator first, Iterator last) : buffer(first, last)
    {
//        std::cout << "non-trivial case" << std::endl;
    }

    iterator_type begin() { return buffer.begin(); }
    iterator_type end()   { return buffer.end(); }
};

template <typename Iterator>
struct trivial_sequence : public detail::_trivial_sequence<Iterator, typename thrust::detail::is_trivial_iterator<Iterator>::type>
{
    typedef _trivial_sequence<Iterator, typename thrust::detail::is_trivial_iterator<Iterator>::type> super_t;

    trivial_sequence(Iterator first, Iterator last) : super_t(first, last) { }
};

} // end namespace detail

} // end namespace thrust

