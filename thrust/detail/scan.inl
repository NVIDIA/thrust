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


/*! \file scan.inl
 *  \brief Inline file for scan.h.
 */

#include <thrust/detail/backend/scan.h>
#include <thrust/detail/backend/generic/select_system.h>
#include <thrust/detail/backend/generic/scan.h>
#include <thrust/iterator/iterator_traits.h>

// XXX make the backend-specific versions of scan available
// XXX try to eliminate the need for these
#include <thrust/detail/backend/cpp/scan.h>
#include <thrust/detail/backend/cuda/scan.h>

#include <thrust/functional.h>

namespace thrust
{

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator inclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::inclusive_scan;

  typedef typename thrust::iterator_space<InputIterator>::type  space1;
  typedef typename thrust::iterator_space<OutputIterator>::type space2;

  return inclusive_scan(select_system(space1(),space2()), first, last, result);
} // end inclusive_scan()

template<typename InputIterator,
         typename OutputIterator,
         typename BinaryFunction>
  OutputIterator inclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                BinaryFunction binary_op)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::inclusive_scan;

  typedef typename thrust::iterator_space<InputIterator>::type  space1;
  typedef typename thrust::iterator_space<OutputIterator>::type space2;

  return inclusive_scan(select_system(space1(),space2()), first, last, result, binary_op);
} // end inclusive_scan()

template<typename InputIterator,
         typename OutputIterator>
  OutputIterator exclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::exclusive_scan;

  typedef typename thrust::iterator_space<InputIterator>::type  space1;
  typedef typename thrust::iterator_space<OutputIterator>::type space2;

  return exclusive_scan(select_system(space1(),space2()), first, last, result);
} // end exclusive_scan()

template<typename InputIterator,
         typename OutputIterator,
         typename T>
  OutputIterator exclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T init)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::exclusive_scan;

  typedef typename thrust::iterator_space<InputIterator>::type  space1;
  typedef typename thrust::iterator_space<OutputIterator>::type space2;

  return exclusive_scan(select_system(space1(),space2()), first, last, result, init);
} // end exclusive_scan()

template<typename InputIterator,
         typename OutputIterator,
         typename T,
         typename BinaryFunction>
  OutputIterator exclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T init,
                                BinaryFunction binary_op)
{
  using thrust::detail::backend::generic::select_system;
  using thrust::detail::backend::generic::exclusive_scan;

  typedef typename thrust::iterator_space<InputIterator>::type  space1;
  typedef typename thrust::iterator_space<OutputIterator>::type space2;

  return exclusive_scan(select_system(space1(),space2()), first, last, result, init, binary_op);
} // end exclusive_scan()

/////////////////////
// Key-Value Scans //
/////////////////////

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator inclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result)
{
    typedef typename thrust::iterator_traits<InputIterator1>::value_type InputType1;
    return thrust::inclusive_scan_by_key(first1, last1, first2, result, thrust::equal_to<InputType1>());
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryPredicate>
  OutputIterator inclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       BinaryPredicate binary_pred)
{
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;
    return thrust::inclusive_scan_by_key(first1, last1, first2, result, binary_pred, thrust::plus<OutputType>());
}

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
    return thrust::detail::backend::inclusive_scan_by_key(first1, last1, first2, result, binary_pred, binary_op);
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator exclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result)
{
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;
    return thrust::exclusive_scan_by_key(first1, last1, first2, result, OutputType(0));
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T>
  OutputIterator exclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init)
{
    typedef typename thrust::iterator_traits<InputIterator1>::value_type InputType1;
    return thrust::exclusive_scan_by_key(first1, last1, first2, result, init, thrust::equal_to<InputType1>());
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename BinaryPredicate>
  OutputIterator exclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init,
                                       BinaryPredicate binary_pred)
{
    typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;
    return thrust::exclusive_scan_by_key(first1, last1, first2, result, init, binary_pred, thrust::plus<OutputType>());
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
                                       T init,
                                       BinaryPredicate binary_pred,
                                       AssociativeOperator binary_op)
{
    return thrust::detail::backend::exclusive_scan_by_key(first1, last1, first2, result, init, binary_pred, binary_op);
}

} // end namespace thrust

