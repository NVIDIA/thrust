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


#include <thrust/pair.h>
#include <thrust/distance.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <thrust/detail/internal_functional.h>

#include <thrust/detail/device/find.h>

// Contributed by Erich Elsen

namespace thrust
{
namespace detail
{
namespace device
{
namespace generic
{

template <typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
thrust::pair<InputIterator1, InputIterator2> mismatch(InputIterator1 first1,
                                                      InputIterator1 last1,
                                                      InputIterator2 first2,
                                                      BinaryPredicate pred)
{
    typedef typename thrust::iterator_traits<InputIterator1>::difference_type difference_type;
    typedef typename thrust::tuple<bool,difference_type> result_type;
    
    const difference_type n = thrust::distance(first1, last1);

    difference_type offset = thrust::detail::device::find_if
                              (
                               thrust::make_transform_iterator
                                (
                                 thrust::make_zip_iterator(thrust::make_tuple(first1, first2)),
                                 thrust::detail::tuple_equal_to<BinaryPredicate>(pred)
                                ),
                               thrust::make_transform_iterator
                                (
                                 thrust::make_zip_iterator(thrust::make_tuple(first1, first2)),
                                 thrust::detail::tuple_equal_to<BinaryPredicate>(pred)
                                ) + n,
                               thrust::detail::equal_to_value<bool>(false)
                              )
                             -
                            thrust::make_transform_iterator
                             (
                              thrust::make_zip_iterator(thrust::make_tuple(first1, first2)),
                              thrust::detail::tuple_equal_to<BinaryPredicate>(pred)
                             );

    return thrust::make_pair(first1 + offset, first2 + offset);
}

} // end namespace generic
} // end namespace device
} // end namespace detail
} // end namespace thrust

