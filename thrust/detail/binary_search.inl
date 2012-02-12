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


/*! \file binary_search.inl
 *  \brief Inline file for binary_search.h.
 */

#include <thrust/detail/config.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/binary_search.h>
#include <thrust/detail/adl_helper.h>

namespace thrust
{

//////////////////////
// Scalar Functions //
//////////////////////

template <class ForwardIterator, class LessThanComparable>
ForwardIterator lower_bound(ForwardIterator first, 
                            ForwardIterator last,
                            const LessThanComparable& value)
{
    using thrust::system::detail::generic::select_system;
    using thrust::system::detail::generic::lower_bound;

    typedef typename thrust::iterator_system<ForwardIterator>::type system; 

    return lower_bound(select_system(system()), first, last, value);
}

template <class ForwardIterator, class T, class StrictWeakOrdering>
ForwardIterator lower_bound(ForwardIterator first,
                            ForwardIterator last,
                            const T& value, 
                            StrictWeakOrdering comp)
{
    using thrust::system::detail::generic::select_system;
    using thrust::system::detail::generic::lower_bound;

    typedef typename thrust::iterator_system<ForwardIterator>::type system; 

    return lower_bound(select_system(system()), first, last, value, comp);
}

template <class ForwardIterator, class LessThanComparable>
ForwardIterator upper_bound(ForwardIterator first, 
                            ForwardIterator last,
                            const LessThanComparable& value)
{
    using thrust::system::detail::generic::select_system;
    using thrust::system::detail::generic::upper_bound;

    typedef typename thrust::iterator_system<ForwardIterator>::type system;

    return upper_bound(select_system(system()), first, last, value);
}

template <class ForwardIterator, class T, class StrictWeakOrdering>
ForwardIterator upper_bound(ForwardIterator first,
                            ForwardIterator last,
                            const T& value, 
                            StrictWeakOrdering comp)
{
    using thrust::system::detail::generic::select_system;
    using thrust::system::detail::generic::upper_bound;

    typedef typename thrust::iterator_system<ForwardIterator>::type system;

    return upper_bound(select_system(system()), first, last, value, comp);
}

template <class ForwardIterator, class LessThanComparable>
bool binary_search(ForwardIterator first, 
                   ForwardIterator last,
                   const LessThanComparable& value)
{
    using thrust::system::detail::generic::select_system;
    using thrust::system::detail::generic::binary_search;

    typedef typename thrust::iterator_system<ForwardIterator>::type system;

    return binary_search(select_system(system()), first, last, value);
}

template <class ForwardIterator, class T, class StrictWeakOrdering>
bool binary_search(ForwardIterator first,
                   ForwardIterator last,
                   const T& value, 
                   StrictWeakOrdering comp)
{
    using thrust::system::detail::generic::select_system;
    using thrust::system::detail::generic::binary_search;

    typedef typename thrust::iterator_system<ForwardIterator>::type system;

    return binary_search(select_system(system()), first, last, value, comp);
}

template <class ForwardIterator, class LessThanComparable>
thrust::pair<ForwardIterator, ForwardIterator>
equal_range(ForwardIterator first,
            ForwardIterator last,
            const LessThanComparable& value)
{
    using thrust::system::detail::generic::select_system;
    using thrust::system::detail::generic::equal_range;

    typedef typename thrust::iterator_system<ForwardIterator>::type system;

    return equal_range(select_system(system()), first, last, value);
}

template <class ForwardIterator, class T, class StrictWeakOrdering>
thrust::pair<ForwardIterator, ForwardIterator>
equal_range(ForwardIterator first,
            ForwardIterator last,
            const T& value,
            StrictWeakOrdering comp)
{
    using thrust::system::detail::generic::select_system;
    using thrust::system::detail::generic::equal_range;

    typedef typename thrust::iterator_system<ForwardIterator>::type system;

    return equal_range(select_system(system()), first, last, value, comp);
}

//////////////////////
// Vector Functions //
//////////////////////

template <class ForwardIterator, class InputIterator, class OutputIterator>
OutputIterator lower_bound(ForwardIterator first, 
                           ForwardIterator last,
                           InputIterator values_first, 
                           InputIterator values_last,
                           OutputIterator output)
{
    using thrust::system::detail::generic::select_system;
    using thrust::system::detail::generic::lower_bound;

    typedef typename thrust::iterator_system<ForwardIterator>::type system1;
    typedef typename thrust::iterator_system<InputIterator>::type   system2;
    typedef typename thrust::iterator_system<OutputIterator>::type  system3;

    return lower_bound(select_system(system1(),system2(),system3()), first, last, values_first, values_last, output);
}

template <class ForwardIterator, class InputIterator, class OutputIterator, class StrictWeakOrdering>
OutputIterator lower_bound(ForwardIterator first, 
                           ForwardIterator last,
                           InputIterator values_first, 
                           InputIterator values_last,
                           OutputIterator output,
                           StrictWeakOrdering comp)
{
    using thrust::system::detail::generic::select_system;
    using thrust::system::detail::generic::lower_bound;

    typedef typename thrust::iterator_system<ForwardIterator>::type system1;
    typedef typename thrust::iterator_system<InputIterator>::type   system2;
    typedef typename thrust::iterator_system<OutputIterator>::type  system3;

    return lower_bound(select_system(system1(),system2(),system3()), first, last, values_first, values_last, output, comp);
}
    
template <class ForwardIterator, class InputIterator, class OutputIterator>
OutputIterator upper_bound(ForwardIterator first, 
                           ForwardIterator last,
                           InputIterator values_first, 
                           InputIterator values_last,
                           OutputIterator output)
{
    using thrust::system::detail::generic::select_system;
    using thrust::system::detail::generic::upper_bound;

    typedef typename thrust::iterator_system<ForwardIterator>::type system1;
    typedef typename thrust::iterator_system<InputIterator>::type   system2;
    typedef typename thrust::iterator_system<OutputIterator>::type  system3;

    return upper_bound(select_system(system1(),system2(),system3()), first, last, values_first, values_last, output);
}

template <class ForwardIterator, class InputIterator, class OutputIterator, class StrictWeakOrdering>
OutputIterator upper_bound(ForwardIterator first, 
                           ForwardIterator last,
                           InputIterator values_first, 
                           InputIterator values_last,
                           OutputIterator output,
                           StrictWeakOrdering comp)
{
    using thrust::system::detail::generic::select_system;
    using thrust::system::detail::generic::upper_bound;

    typedef typename thrust::iterator_system<ForwardIterator>::type system1;
    typedef typename thrust::iterator_system<InputIterator>::type   system2;
    typedef typename thrust::iterator_system<OutputIterator>::type  system3;

    return upper_bound(select_system(system1(),system2(),system3()), first, last, values_first, values_last, output, comp);
}

template <class ForwardIterator, class InputIterator, class OutputIterator>
OutputIterator binary_search(ForwardIterator first, 
                             ForwardIterator last,
                             InputIterator values_first, 
                             InputIterator values_last,
                             OutputIterator output)
{
    using thrust::system::detail::generic::select_system;
    using thrust::system::detail::generic::binary_search;

    typedef typename thrust::iterator_system<ForwardIterator>::type system1;
    typedef typename thrust::iterator_system<InputIterator>::type   system2;
    typedef typename thrust::iterator_system<OutputIterator>::type  system3;

    return binary_search(select_system(system1(),system2(),system3()), first, last, values_first, values_last, output);
}

template <class ForwardIterator, class InputIterator, class OutputIterator, class StrictWeakOrdering>
OutputIterator binary_search(ForwardIterator first, 
                             ForwardIterator last,
                             InputIterator values_first, 
                             InputIterator values_last,
                             OutputIterator output,
                             StrictWeakOrdering comp)
{
    using thrust::system::detail::generic::select_system;
    using thrust::system::detail::generic::binary_search;

    typedef typename thrust::iterator_system<ForwardIterator>::type system1;
    typedef typename thrust::iterator_system<InputIterator>::type   system2;
    typedef typename thrust::iterator_system<OutputIterator>::type  system3;

    return binary_search(select_system(system1(),system2(),system3()), first, last, values_first, values_last, output, comp);
}

} // end namespace thrust

