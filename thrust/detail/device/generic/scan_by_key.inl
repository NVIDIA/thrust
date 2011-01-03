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


#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/replace.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/raw_buffer.h>
#include <thrust/detail/internal_functional.h>

#include <thrust/detail/device/scan.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace generic
{
namespace detail
{

template <typename OutputType, typename HeadFlagType, typename AssociativeOperator>
struct segmented_scan_functor
{
    AssociativeOperator binary_op;

    typedef typename thrust::tuple<OutputType, HeadFlagType> result_type;

    __host__ __device__
    segmented_scan_functor(AssociativeOperator _binary_op) : binary_op(_binary_op) {}

    __host__ __device__
    result_type operator()(result_type a, result_type b)
    {
        return result_type(thrust::get<1>(b) ? thrust::get<0>(b) : binary_op(thrust::get<0>(a), thrust::get<0>(b)),
                           thrust::get<1>(a) | thrust::get<1>(b));
    }
};

} // end namespace detail

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryPredicate,
         typename AssociativeOperator>
  OutputIterator inclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       BinaryPredicate binary_pred,
                                       AssociativeOperator binary_op)
{
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;
    typedef typename thrust::iterator_space<OutputIterator>::type Space;
    typedef unsigned int HeadFlagType;

    const size_t n = last1 - first1;

    if(n != 0)
    {
        // compute head flags
        thrust::detail::raw_buffer<HeadFlagType,Space> flags(n);
        flags[0] = 1; thrust::transform(first1, last1 - 1, first1 + 1, flags.begin() + 1, thrust::detail::not2(binary_pred));

        // scan key-flag tuples, 
        // For additional details refer to Section 2 of the following paper
        //    S. Sengupta, M. Harris, and M. Garland. "Efficient parallel scan algorithms for GPUs"
        //    NVIDIA Technical Report NVR-2008-003, December 2008
        //    http://mgarland.org/files/papers/nvr-2008-003.pdf
        thrust::detail::device::inclusive_scan
            (thrust::make_zip_iterator(thrust::make_tuple(first2, flags.begin())),
             thrust::make_zip_iterator(thrust::make_tuple(first2, flags.begin())) + n,
             thrust::make_zip_iterator(thrust::make_tuple(result, flags.begin())),
             detail::segmented_scan_functor<OutputType, HeadFlagType, AssociativeOperator>(binary_op));
    }

    return result + n;
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename BinaryPredicate,
         typename AssociativeOperator>
  OutputIterator exclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       const T init,
                                       BinaryPredicate binary_pred,
                                       AssociativeOperator binary_op)
{
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;
    typedef typename thrust::iterator_space<OutputIterator>::type        Space;
    typedef unsigned int HeadFlagType;

    const size_t n = last1 - first1;

    if(n != 0)
    {
        InputIterator2 last2 = first2 + n;

        // compute head flags
        thrust::detail::raw_buffer<HeadFlagType,Space> flags(n);
        flags[0] = 1; thrust::transform(first1, last1 - 1, first1 + 1, flags.begin() + 1, thrust::detail::not2(binary_pred));

        // shift input one to the right and initialize segments with init
        thrust::detail::raw_buffer<OutputType,Space> temp(n);
        thrust::replace_copy_if(first2, last2 - 1, flags.begin() + 1, temp.begin() + 1, thrust::negate<HeadFlagType>(), init);
        temp[0] = init;

        // scan key-flag tuples, 
        // For additional details refer to Section 2 of the following paper
        //    S. Sengupta, M. Harris, and M. Garland. "Efficient parallel scan algorithms for GPUs"
        //    NVIDIA Technical Report NVR-2008-003, December 2008
        //    http://mgarland.org/files/papers/nvr-2008-003.pdf
        thrust::detail::device::inclusive_scan(thrust::make_zip_iterator(thrust::make_tuple(temp.begin(), flags.begin())),
                                               thrust::make_zip_iterator(thrust::make_tuple(temp.begin(), flags.begin())) + n,
                                               thrust::make_zip_iterator(thrust::make_tuple(result,       flags.begin())),
                                               detail::segmented_scan_functor<OutputType, HeadFlagType, AssociativeOperator>(binary_op));
    }

    return result + n;
}

} // end namespace generic
} // end namespace device
} // end namespace detail
} // end namespace thrust

