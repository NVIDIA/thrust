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
#include <thrust/system/detail/adl/binary_search.h>

namespace thrust
{


template <typename System, typename ForwardIterator, typename LessThanComparable>
ForwardIterator lower_bound(thrust::detail::dispatchable_base<System> &system,
                            ForwardIterator first,
                            ForwardIterator last,
                            const LessThanComparable &value)
{
    using thrust::system::detail::generic::lower_bound;
    return lower_bound(system.derived(), first, last, value);
}


template<typename System, typename ForwardIterator, typename T, typename StrictWeakOrdering>
ForwardIterator lower_bound(thrust::detail::dispatchable_base<System> &system,
                            ForwardIterator first,
                            ForwardIterator last,
                            const T &value,
                            StrictWeakOrdering comp)
{
    using thrust::system::detail::generic::lower_bound;
    return lower_bound(system.derived(), first, last, value, comp);
}


template<typename System, typename ForwardIterator, typename LessThanComparable>
ForwardIterator upper_bound(thrust::detail::dispatchable_base<System> &system,
                            ForwardIterator first,
                            ForwardIterator last,
                            const LessThanComparable &value)
{
    using thrust::system::detail::generic::upper_bound;
    return upper_bound(system.derived(), first, last, value);
}


template<typename System, typename ForwardIterator, typename T, typename StrictWeakOrdering>
ForwardIterator upper_bound(thrust::detail::dispatchable_base<System> &system,
                            ForwardIterator first,
                            ForwardIterator last,
                            const T &value,
                            StrictWeakOrdering comp)
{
    using thrust::system::detail::generic::upper_bound;
    return upper_bound(system.derived(), first, last, value, comp);
}


template <typename System, typename ForwardIterator, typename LessThanComparable>
bool binary_search(thrust::detail::dispatchable_base<System> &system,
                   ForwardIterator first, 
                   ForwardIterator last,
                   const LessThanComparable& value)
{
    using thrust::system::detail::generic::binary_search;
    return binary_search(system.derived(), first, last, value);
}


template <typename System, typename ForwardIterator, typename T, typename StrictWeakOrdering>
bool binary_search(thrust::detail::dispatchable_base<System> &system,
                   ForwardIterator first,
                   ForwardIterator last,
                   const T& value, 
                   StrictWeakOrdering comp)
{
    using thrust::system::detail::generic::binary_search;
    return binary_search(system.derived(), first, last, value, comp);
}


template <typename System, typename ForwardIterator, typename T, typename StrictWeakOrdering>
thrust::pair<ForwardIterator, ForwardIterator>
equal_range(thrust::detail::dispatchable_base<System> &system,
            ForwardIterator first,
            ForwardIterator last,
            const T& value,
            StrictWeakOrdering comp)
{
    using thrust::system::detail::generic::equal_range;
    return equal_range(system.derived(), first, last, value, comp);
}


template <typename System, typename ForwardIterator, typename LessThanComparable>
thrust::pair<ForwardIterator, ForwardIterator>
equal_range(thrust::detail::dispatchable_base<System> &system,
            ForwardIterator first,
            ForwardIterator last,
            const LessThanComparable& value)
{
    using thrust::system::detail::generic::equal_range;
    return equal_range(system.derived(), first, last, value);
}


template <typename System, typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator lower_bound(thrust::detail::dispatchable_base<System> &system,
                           ForwardIterator first, 
                           ForwardIterator last,
                           InputIterator values_first, 
                           InputIterator values_last,
                           OutputIterator output)
{
    using thrust::system::detail::generic::lower_bound;
    return lower_bound(system.derived(), first, last, values_first, values_last, output);
}


template <typename System, typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
OutputIterator lower_bound(thrust::detail::dispatchable_base<System> &system,
                           ForwardIterator first, 
                           ForwardIterator last,
                           InputIterator values_first, 
                           InputIterator values_last,
                           OutputIterator output,
                           StrictWeakOrdering comp)
{
    using thrust::system::detail::generic::lower_bound;
    return lower_bound(system.derived(), first, last, values_first, values_last, output, comp);
}


template <typename System, typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator upper_bound(thrust::detail::dispatchable_base<System> &system,
                           ForwardIterator first, 
                           ForwardIterator last,
                           InputIterator values_first, 
                           InputIterator values_last,
                           OutputIterator output)
{
    using thrust::system::detail::generic::upper_bound;
    return upper_bound(system.derived(), first, last, values_first, values_last, output);
}


template <typename System, typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
OutputIterator upper_bound(thrust::detail::dispatchable_base<System> &system,
                           ForwardIterator first, 
                           ForwardIterator last,
                           InputIterator values_first, 
                           InputIterator values_last,
                           OutputIterator output,
                           StrictWeakOrdering comp)
{
    using thrust::system::detail::generic::upper_bound;
    return upper_bound(system.derived(), first, last, values_first, values_last, output, comp);
}


template <typename System, typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator binary_search(thrust::detail::dispatchable_base<System> &system,
                             ForwardIterator first, 
                             ForwardIterator last,
                             InputIterator values_first, 
                             InputIterator values_last,
                             OutputIterator output)
{
    using thrust::system::detail::generic::binary_search;
    return binary_search(system.derived(), first, last, values_first, values_last, output);
}


template <typename System, typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
OutputIterator binary_search(thrust::detail::dispatchable_base<System> &system,
                             ForwardIterator first, 
                             ForwardIterator last,
                             InputIterator values_first, 
                             InputIterator values_last,
                             OutputIterator output,
                             StrictWeakOrdering comp)
{
    using thrust::system::detail::generic::binary_search;
    return binary_search(system.derived(), first, last, values_first, values_last, output, comp);
}


namespace detail
{


template <typename System, typename ForwardIterator, typename LessThanComparable>
ForwardIterator strip_const_lower_bound(const System &system,
                                        ForwardIterator first,
                                        ForwardIterator last,
                                        const LessThanComparable &value)
{
    System &non_const_system = const_cast<System&>(system);
    return thrust::lower_bound(non_const_system, first, last, value);
}


template <typename System, typename ForwardIterator, typename T, typename StrictWeakOrdering>
ForwardIterator strip_const_lower_bound(const System &system,
                                        ForwardIterator first,
                                        ForwardIterator last,
                                        const T &value,
                                        StrictWeakOrdering comp)
{
    System &non_const_system = const_cast<System&>(system);
    return thrust::lower_bound(non_const_system, first, last, value, comp);
}


template <typename System, typename ForwardIterator, typename LessThanComparable>
ForwardIterator strip_const_upper_bound(const System &system,
                                        ForwardIterator first,
                                        ForwardIterator last,
                                        const LessThanComparable &value)
{
    System &non_const_system = const_cast<System&>(system);
    return thrust::upper_bound(non_const_system, first, last, value);
}


template <typename System, typename ForwardIterator, typename T, typename StrictWeakOrdering>
ForwardIterator strip_const_upper_bound(const System &system,
                                        ForwardIterator first,
                                        ForwardIterator last,
                                        const T &value,
                                        StrictWeakOrdering comp)
{
    System &non_const_system = const_cast<System&>(system);
    return thrust::upper_bound(non_const_system, first, last, value, comp);
}


template <typename System, typename ForwardIterator, typename LessThanComparable>
bool strip_const_binary_search(const System &system,
                               ForwardIterator first, 
                               ForwardIterator last,
                               const LessThanComparable& value)
{
    System &non_const_system = const_cast<System&>(system);
    return thrust::binary_search(non_const_system, first, last, value);
}


template <typename System, typename ForwardIterator, typename T, typename StrictWeakOrdering>
bool strip_const_binary_search(const System &system,
                               ForwardIterator first,
                               ForwardIterator last,
                               const T& value, 
                               StrictWeakOrdering comp)
{
    System &non_const_system = const_cast<System&>(system);
    return thrust::binary_search(non_const_system, first, last, value, comp);
}


template <typename System, typename ForwardIterator, typename T, typename StrictWeakOrdering>
thrust::pair<ForwardIterator, ForwardIterator>
strip_const_equal_range(const System &system,
                        ForwardIterator first,
                        ForwardIterator last,
                        const T& value,
                        StrictWeakOrdering comp)
{
    System &non_const_system = const_cast<System&>(system);
    return thrust::equal_range(non_const_system, first, last, value, comp);
}


template <typename System, typename ForwardIterator, typename LessThanComparable>
thrust::pair<ForwardIterator, ForwardIterator>
strip_const_equal_range(const System &system,
                        ForwardIterator first,
                        ForwardIterator last,
                        const LessThanComparable& value)
{
    System &non_const_system = const_cast<System&>(system);
    return thrust::equal_range(non_const_system, first, last, value);
}


template <typename System, typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator strip_const_lower_bound(const System &system,
                                       ForwardIterator first, 
                                       ForwardIterator last,
                                       InputIterator values_first, 
                                       InputIterator values_last,
                                       OutputIterator output)
{
    System &non_const_system = const_cast<System&>(system);
    return thrust::lower_bound(non_const_system, first, last, values_first, values_last, output);
}


template <typename System, typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
OutputIterator strip_const_lower_bound(const System &system,
                                       ForwardIterator first, 
                                       ForwardIterator last,
                                       InputIterator values_first, 
                                       InputIterator values_last,
                                       OutputIterator output,
                                       StrictWeakOrdering comp)
{
    System &non_const_system = const_cast<System&>(system);
    return thrust::lower_bound(non_const_system, first, last, values_first, values_last, output, comp);
}


template <typename System, typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator strip_const_upper_bound(const System &system,
                                       ForwardIterator first, 
                                       ForwardIterator last,
                                       InputIterator values_first, 
                                       InputIterator values_last,
                                       OutputIterator output)
{
    System &non_const_system = const_cast<System&>(system);
    return thrust::upper_bound(non_const_system, first, last, values_first, values_last, output);
}


template <typename System, typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
OutputIterator strip_const_upper_bound(const System &system,
                                       ForwardIterator first, 
                                       ForwardIterator last,
                                       InputIterator values_first, 
                                       InputIterator values_last,
                                       OutputIterator output,
                                       StrictWeakOrdering comp)
{
    System &non_const_system = const_cast<System&>(system);
    return thrust::upper_bound(non_const_system, first, last, values_first, values_last, output, comp);
}


template <typename System, typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator strip_const_binary_search(const System &system,
                                         ForwardIterator first, 
                                         ForwardIterator last,
                                         InputIterator values_first, 
                                         InputIterator values_last,
                                         OutputIterator output)
{
    System &non_const_system = const_cast<System&>(system);
    return thrust::binary_search(non_const_system, first, last, values_first, values_last, output);
}


template <typename System, typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
OutputIterator strip_const_binary_search(const System &system,
                                         ForwardIterator first, 
                                         ForwardIterator last,
                                         InputIterator values_first, 
                                         InputIterator values_last,
                                         OutputIterator output,
                                         StrictWeakOrdering comp)
{
    System &non_const_system = const_cast<System&>(system);
    return thrust::binary_search(non_const_system, first, last, values_first, values_last, output, comp);
}


} // end detail


//////////////////////
// Scalar Functions //
//////////////////////

template <typename ForwardIterator, typename LessThanComparable>
ForwardIterator lower_bound(ForwardIterator first, 
                            ForwardIterator last,
                            const LessThanComparable& value)
{
    using thrust::system::detail::generic::select_system;

    typedef typename thrust::iterator_system<ForwardIterator>::type system; 

    return thrust::detail::strip_const_lower_bound(select_system(system()), first, last, value);
}

template <typename ForwardIterator, typename T, typename StrictWeakOrdering>
ForwardIterator lower_bound(ForwardIterator first,
                            ForwardIterator last,
                            const T& value, 
                            StrictWeakOrdering comp)
{
    using thrust::system::detail::generic::select_system;

    typedef typename thrust::iterator_system<ForwardIterator>::type system; 

    return thrust::detail::strip_const_lower_bound(select_system(system()), first, last, value, comp);
}

template <typename ForwardIterator, typename LessThanComparable>
ForwardIterator upper_bound(ForwardIterator first, 
                            ForwardIterator last,
                            const LessThanComparable& value)
{
    using thrust::system::detail::generic::select_system;

    typedef typename thrust::iterator_system<ForwardIterator>::type system;

    return thrust::detail::strip_const_upper_bound(select_system(system()), first, last, value);
}

template <typename ForwardIterator, typename T, typename StrictWeakOrdering>
ForwardIterator upper_bound(ForwardIterator first,
                            ForwardIterator last,
                            const T& value, 
                            StrictWeakOrdering comp)
{
    using thrust::system::detail::generic::select_system;

    typedef typename thrust::iterator_system<ForwardIterator>::type system;

    return thrust::detail::strip_const_upper_bound(select_system(system()), first, last, value, comp);
}

template <typename ForwardIterator, typename LessThanComparable>
bool binary_search(ForwardIterator first, 
                   ForwardIterator last,
                   const LessThanComparable& value)
{
    using thrust::system::detail::generic::select_system;

    typedef typename thrust::iterator_system<ForwardIterator>::type system;

    return thrust::detail::strip_const_binary_search(select_system(system()), first, last, value);
}

template <typename ForwardIterator, typename T, typename StrictWeakOrdering>
bool binary_search(ForwardIterator first,
                   ForwardIterator last,
                   const T& value, 
                   StrictWeakOrdering comp)
{
    using thrust::system::detail::generic::select_system;

    typedef typename thrust::iterator_system<ForwardIterator>::type system;

    return thrust::detail::strip_const_binary_search(select_system(system()), first, last, value, comp);
}

template <typename ForwardIterator, typename LessThanComparable>
thrust::pair<ForwardIterator, ForwardIterator>
equal_range(ForwardIterator first,
            ForwardIterator last,
            const LessThanComparable& value)
{
    using thrust::system::detail::generic::select_system;

    typedef typename thrust::iterator_system<ForwardIterator>::type system;

    return thrust::detail::strip_const_equal_range(select_system(system()), first, last, value);
}

template <typename ForwardIterator, typename T, typename StrictWeakOrdering>
thrust::pair<ForwardIterator, ForwardIterator>
equal_range(ForwardIterator first,
            ForwardIterator last,
            const T& value,
            StrictWeakOrdering comp)
{
    using thrust::system::detail::generic::select_system;

    typedef typename thrust::iterator_system<ForwardIterator>::type system;

    return thrust::detail::strip_const_equal_range(select_system(system()), first, last, value, comp);
}

//////////////////////
// Vector Functions //
//////////////////////

template <typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator lower_bound(ForwardIterator first, 
                           ForwardIterator last,
                           InputIterator values_first, 
                           InputIterator values_last,
                           OutputIterator output)
{
    using thrust::system::detail::generic::select_system;

    typedef typename thrust::iterator_system<ForwardIterator>::type system1;
    typedef typename thrust::iterator_system<InputIterator>::type   system2;
    typedef typename thrust::iterator_system<OutputIterator>::type  system3;

    return thrust::detail::strip_const_lower_bound(select_system(system1(),system2(),system3()), first, last, values_first, values_last, output);
}

template <typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
OutputIterator lower_bound(ForwardIterator first, 
                           ForwardIterator last,
                           InputIterator values_first, 
                           InputIterator values_last,
                           OutputIterator output,
                           StrictWeakOrdering comp)
{
    using thrust::system::detail::generic::select_system;

    typedef typename thrust::iterator_system<ForwardIterator>::type system1;
    typedef typename thrust::iterator_system<InputIterator>::type   system2;
    typedef typename thrust::iterator_system<OutputIterator>::type  system3;

    return thrust::detail::strip_const_lower_bound(select_system(system1(),system2(),system3()), first, last, values_first, values_last, output, comp);
}
    
template <typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator upper_bound(ForwardIterator first, 
                           ForwardIterator last,
                           InputIterator values_first, 
                           InputIterator values_last,
                           OutputIterator output)
{
    using thrust::system::detail::generic::select_system;

    typedef typename thrust::iterator_system<ForwardIterator>::type system1;
    typedef typename thrust::iterator_system<InputIterator>::type   system2;
    typedef typename thrust::iterator_system<OutputIterator>::type  system3;

    return thrust::detail::strip_const_upper_bound(select_system(system1(),system2(),system3()), first, last, values_first, values_last, output);
}

template <typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
OutputIterator upper_bound(ForwardIterator first, 
                           ForwardIterator last,
                           InputIterator values_first, 
                           InputIterator values_last,
                           OutputIterator output,
                           StrictWeakOrdering comp)
{
    using thrust::system::detail::generic::select_system;

    typedef typename thrust::iterator_system<ForwardIterator>::type system1;
    typedef typename thrust::iterator_system<InputIterator>::type   system2;
    typedef typename thrust::iterator_system<OutputIterator>::type  system3;

    return thrust::detail::strip_const_upper_bound(select_system(system1(),system2(),system3()), first, last, values_first, values_last, output, comp);
}

template <typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator binary_search(ForwardIterator first, 
                             ForwardIterator last,
                             InputIterator values_first, 
                             InputIterator values_last,
                             OutputIterator output)
{
    using thrust::system::detail::generic::select_system;

    typedef typename thrust::iterator_system<ForwardIterator>::type system1;
    typedef typename thrust::iterator_system<InputIterator>::type   system2;
    typedef typename thrust::iterator_system<OutputIterator>::type  system3;

    return thrust::detail::strip_const_binary_search(select_system(system1(),system2(),system3()), first, last, values_first, values_last, output);
}

template <typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
OutputIterator binary_search(ForwardIterator first, 
                             ForwardIterator last,
                             InputIterator values_first, 
                             InputIterator values_last,
                             OutputIterator output,
                             StrictWeakOrdering comp)
{
    using thrust::system::detail::generic::select_system;

    typedef typename thrust::iterator_system<ForwardIterator>::type system1;
    typedef typename thrust::iterator_system<InputIterator>::type   system2;
    typedef typename thrust::iterator_system<OutputIterator>::type  system3;

    return thrust::detail::strip_const_binary_search(select_system(system1(),system2(),system3()), first, last, values_first, values_last, output, comp);
}

} // end namespace thrust

