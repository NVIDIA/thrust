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

#include <thrust/detail/device/reduce.h>

#include <thrust/functional.h>
#include <thrust/tuple.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

// Implementation of find_if() using short-circuiting
// Contributed by Erich Elsen

namespace thrust
{
namespace detail
{
namespace device
{
namespace generic
{

template <typename InputIterator, typename Predicate>
InputIterator find_if(InputIterator first,
                      InputIterator last,
                      Predicate pred)
{
    typedef typename thrust::iterator_traits<InputIterator>::difference_type difference_type;
    typedef typename thrust::tuple<bool,difference_type> result_type;
   
    // empty sequence
    if (first == last)
        return last;

    const difference_type n = thrust::distance(first, last);

    // TODO incorporate sizeof(InputType) into interval_threshold and round to multiple of 32
    const difference_type interval_threshold = n; //1 << 20; // XXX disabled until performance is sorted out
    const difference_type interval_size = std::min(interval_threshold, n);

    for(difference_type begin = 0; begin < n; begin += interval_size)
    {
        difference_type end = thrust::min(begin + interval_size, n);

        result_type result = thrust::detail::device::reduce
            (
             thrust::make_zip_iterator
             (
              thrust::make_tuple
              (
               thrust::transform_iterator<Predicate, InputIterator, bool>(first, pred),  // note: must specify Reference=bool
               thrust::counting_iterator<difference_type>(0)
              )
             ) + begin,
             thrust::make_zip_iterator
             (
              thrust::make_tuple
              (
               thrust::transform_iterator<Predicate, InputIterator, bool>(first, pred),  // note: must specify Reference=bool
               thrust::counting_iterator<difference_type>(0)
              )
             ) + end,
             result_type(false, end),
             thrust::maximum<result_type>()
            );

        // see if we found something
        if (thrust::get<1>(result) != end)
        {
            return first + thrust::get<1>(result);
        }
    }

    //nothing was found if we reach here...
    return last;
}

} // end namespace generic
} // end namespace device
} // end namespace detail
} // end namespace thrust

