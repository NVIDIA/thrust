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


#include <thrust/detail/backend/cpp/sort.h>

#include <thrust/iterator/detail/forced_iterator.h> // XXX remove this we we have a proper OMP sort
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace omp
{

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
void stable_sort(RandomAccessIterator first,
                 RandomAccessIterator last,
                 StrictWeakOrdering comp)
{
    // XXX use cpp stable_sort implementation for now
    thrust::detail::backend::cpp::stable_sort(thrust::detail::make_forced_iterator(first,   thrust::host_space_tag()),
                                              thrust::detail::make_forced_iterator(last,    thrust::host_space_tag()),
                                              comp);
}

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
void stable_sort_by_key(RandomAccessIterator1 keys_first,
                        RandomAccessIterator1 keys_last,
                        RandomAccessIterator2 values_first,
                        StrictWeakOrdering comp)
{
    // XXX use cpp stable_sort_by_key implementation for now
    thrust::detail::backend::cpp::stable_sort_by_key(thrust::detail::make_forced_iterator(keys_first,   thrust::host_space_tag()),
                                                     thrust::detail::make_forced_iterator(keys_last,    thrust::host_space_tag()),
                                                     thrust::detail::make_forced_iterator(values_first, thrust::host_space_tag()),
                                                     comp);
}

} // end namespace omp
} // end namespace backend
} // end namespace detail
} // end namespace thrust

