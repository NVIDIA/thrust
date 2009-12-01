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
 *  WIKeyTypeHOUKeyType WARRANKeyTypeIES OR CONDIKeyTypeIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


/*! \file indirect_stable_sort.inl
 *  \brief Inline file for indirect_stable_sort.h
 */

#pragma once

#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/device/cuda/sort.h>

#include <thrust/detail/raw_buffer.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace cuda
{
namespace detail
{
    // add one level of indirection to the StrictWeakOrdering comp
    template <typename RandomAccessIterator, typename StrictWeakOrdering> 
    struct indirect_comp
    {
        RandomAccessIterator first;
        StrictWeakOrdering   comp;
    
        indirect_comp(RandomAccessIterator first, StrictWeakOrdering comp)
            : first(first), comp(comp) {}
    
        template <typename IndexKeyTypeype>
        __host__ __device__
        bool operator()(IndexKeyTypeype a, IndexKeyTypeype b)
        {
            return comp(thrust::detail::device::dereference(first, a),
                        thrust::detail::device::dereference(first, b));
        }    
    };


    
template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
void indirect_stable_sort(RandomAccessIterator first,
                          RandomAccessIterator last,
                          StrictWeakOrdering comp)
{
    typedef typename thrust::iterator_traits<RandomAccessIterator>::value_type KeyType;

    thrust::detail::raw_device_buffer<unsigned int> permutation(last - first);
    thrust::sequence(permutation.begin(), permutation.end());

    thrust::detail::device::cuda::detail::stable_merge_sort(permutation.begin(), permutation.end(),
            indirect_comp<RandomAccessIterator,StrictWeakOrdering>(first, comp));

    thrust::detail::raw_device_buffer<KeyType> temp(first, last);

    thrust::gather(first, last, permutation.begin(), temp.begin());
}
    
template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
void indirect_stable_sort_by_key(RandomAccessIterator1 keys_first,
                                 RandomAccessIterator1 keys_last,
                                 RandomAccessIterator2 values_first,
                                 StrictWeakOrdering comp)
{
    typedef typename thrust::iterator_traits<RandomAccessIterator1>::value_type KeyType;

    thrust::detail::raw_device_buffer<unsigned int> permutation(keys_last - keys_first);
    thrust::sequence(permutation.begin(), permutation.end());

    thrust::detail::device::cuda::stable_sort_by_key(permutation.begin(), permutation.end(), values_first,
            indirect_comp<RandomAccessIterator1,StrictWeakOrdering>(keys_first, comp));

    thrust::detail::raw_device_buffer<KeyType> temp(keys_first, keys_last);

    thrust::gather(keys_first, keys_last, permutation.begin(), temp.begin());
}

} // end namespace detail
} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace thrust

