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

//  Copyright Neil Groves 2009.
//  Use, modification and distribution is subject to the Boost Software
//  License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
// For more information, see http://www.boost.org/libs/range/

#pragma once

#include <thrust/detail/config.h>
#include <thrust/range/detail/metafunctions.h>
#include <thrust/range/detail/is_range.h>
#include <cstddef> // for std::size_t

namespace thrust
{

namespace experimental
{

template<typename IteratorOrRange>
  class iterator_range
{
  private:
    typedef iterator_range<IteratorOrRange> this_type;

  public:
    typedef typename thrust::detail::eval_if<
      detail::is_range<IteratorOrRange>::value,
      range_iterator<IteratorOrRange>,
      thrust::detail::identity_<IteratorOrRange>
    >::type                                              iterator;

    typedef typename iterator_value<iterator>::type      value_type;
    typedef typename iterator_difference<iterator>::type difference_type;
    typedef std::size_t                                  size_type; 
    typedef typename iterator_reference<iterator>::type  reference;
    typedef iterator                                     const_iterator;

    __host__ __device__
    inline iterator_range(void);

    // constructor from a pair of iterators
    template<typename Iterator>
    __host__ __device__
    inline iterator_range(Iterator begin, Iterator end);

    // XXX boost uses tag dispatch for a couple of these constructors
    //     do we need to care about that?
    // XXX boost uses detail::adl_begin & detail::adl_end to implement these
    //     do we need to?

    template<typename ForwardRange>
    __host__ __device__
    inline iterator_range(const ForwardRange &r);

    template<typename ForwardRange>
    __host__ __device__
    inline iterator_range(ForwardRange &r);
    
    template<typename Iterator>
    __host__ __device__
    inline iterator_range &operator=(const iterator_range<Iterator> &r);

    template<typename ForwardRange>
    __host__ __device__
    inline iterator_range &operator=(const ForwardRange &r);

    template<typename ForwardRange>
    __host__ __device__
    inline iterator_range &operator=(ForwardRange &r);

    __host__ __device__
    inline iterator begin(void) const;

    __host__ __device__
    inline iterator end(void) const;

    __host__ __device__
    inline difference_type size(void) const;

    __host__ __device__
    inline bool empty(void) const;

    // XXX conversion to unspecified_bool_type here, which seem to depend
    //     on the ability to return a function pointer
    //     use this WAR for now
    __host__ __device__
    inline operator bool (void) const;

    __host__ __device__
    inline bool equal(const iterator_range& r) const;

    __host__ __device__
    inline reference front(void) const;

    __host__ __device__
    inline reference back(void) const;

    __host__ __device__
    inline reference operator[](difference_type at) const;

    // XXX advance_begin & advance_end here

  private:
    iterator m_begin, m_end;
}; // end iterator_range


template<typename Iterator>
  inline iterator_range<Iterator>
    make_iterator_range(Iterator begin, Iterator end);

} // end experimental

} // end thrust

#include <thrust/range/detail/iterator_range.inl>

