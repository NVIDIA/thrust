/*
 *  Copyright 2008-2013 NVIDIA Corporation
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


#include <thrust/detail/config.h>
#include <thrust/system/detail/generic/scan_by_key.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/replace.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/detail/range/head_flags.h>
#include <thrust/scan.h>
#include <thrust/copy.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{
namespace detail
{

// for *_scan_by_key the arguments of the BinaryPredicate for head_flags must be swapped: 
// "consecutive iterators i and i+1 in the range [first1, last1) belong to the same segment if binary_pred(*i, *(i+1)) is true"
template<typename BinaryPredicate>
struct swap_arguments_binary_predicate
{
  BinaryPredicate binary_pred;

  __host__ __device__
  swap_arguments_binary_predicate(BinaryPredicate binary_pred) : binary_pred(binary_pred) {}

  template<typename T1, typename T2>
  __host__ __device__ __thrust_forceinline__
  bool operator()(const T1& lhs, const T2& rhs)
  {
      return binary_pred(rhs, lhs);
  }
}; // end swap_arguments



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


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
__host__ __device__
  OutputIterator inclusive_scan_by_key(thrust::execution_policy<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result)
{
  typedef typename thrust::iterator_traits<InputIterator1>::value_type InputType1;
  return thrust::inclusive_scan_by_key(exec, first1, last1, first2, result, thrust::equal_to<InputType1>());
}


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryPredicate>
__host__ __device__
  OutputIterator inclusive_scan_by_key(thrust::execution_policy<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       BinaryPredicate binary_pred)
{
  typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;
  return thrust::inclusive_scan_by_key(exec, first1, last1, first2, result, binary_pred, thrust::plus<OutputType>());
}


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryPredicate,
         typename AssociativeOperator>
__host__ __device__
  OutputIterator inclusive_scan_by_key(thrust::execution_policy<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       BinaryPredicate binary_pred,
                                       AssociativeOperator binary_op)
{
  typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;
  typedef unsigned int HeadFlagType;

  const size_t n = last1 - first1;

  if(n != 0)
  {
    // compute head flags
    typedef detail::swap_arguments_binary_predicate<BinaryPredicate> SwappedBinaryPredicate;
    thrust::detail::head_flags<InputIterator1, SwappedBinaryPredicate> head_flags(first1, last1, SwappedBinaryPredicate(binary_pred));

    // scan key-flag tuples, 
    // For additional details refer to Section 2 of the following paper
    //    S. Sengupta, M. Harris, and M. Garland. "Efficient parallel scan algorithms for GPUs"
    //    NVIDIA Technical Report NVR-2008-003, December 2008
    //    http://mgarland.org/files/papers/nvr-2008-003.pdf
    thrust::inclusive_scan(exec,
                           thrust::make_zip_iterator(thrust::make_tuple(first2, head_flags.begin())),
                           thrust::make_zip_iterator(thrust::make_tuple(first2, head_flags.begin())) + n,
                           thrust::make_zip_iterator(thrust::make_tuple(result, thrust::make_discard_iterator())),
                           detail::segmented_scan_functor<OutputType, HeadFlagType, AssociativeOperator>(binary_op));
  }

  return result + n;
}


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
__host__ __device__
  OutputIterator exclusive_scan_by_key(thrust::execution_policy<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result)
{
  typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;
  return thrust::exclusive_scan_by_key(exec, first1, last1, first2, result, OutputType(0));
}


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T>
__host__ __device__
  OutputIterator exclusive_scan_by_key(thrust::execution_policy<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init)
{
  typedef typename thrust::iterator_traits<InputIterator1>::value_type InputType1;
  return thrust::exclusive_scan_by_key(exec, first1, last1, first2, result, init, thrust::equal_to<InputType1>());
}


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename BinaryPredicate>
__host__ __device__
  OutputIterator exclusive_scan_by_key(thrust::execution_policy<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init,
                                       BinaryPredicate binary_pred)
{
  typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;
  return thrust::exclusive_scan_by_key(exec, first1, last1, first2, result, init, binary_pred, thrust::plus<OutputType>());
}


template <typename RandomAccessIterator,
          typename ValueType,
          typename IndexType = typename thrust::iterator_difference<RandomAccessIterator>::type>
struct init_functor
{
    typedef ValueType result_type;
    const ValueType init;
    const RandomAccessIterator begin;

    __host__ __device__
    init_functor(RandomAccessIterator begin, const ValueType& init) : begin(begin), init(init) {}

    template <typename Tuple>
    __host__ __device__ __thrust_forceinline__
    result_type operator()(const Tuple& t) const
    {
        // shift input one to the right and initialize segments with init
        if (thrust::get<1>(t))
        {
            return init;
        }
        const IndexType i = thrust::get<0>(t);
        return *(begin + i - 1);
    }
};


template<typename Iterator>
__host__ __device__
  bool iterator_equal(const Iterator& it1, const Iterator& it2)
{
    return it1 == it2;
}

template<typename Iterator1, typename Iterator2>
__host__ __device__
  bool iterator_equal(const Iterator1&, const Iterator2&)
{
    return false;
}

template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename BinaryPredicate,
         typename AssociativeOperator>
__host__ __device__
  OutputIterator exclusive_scan_by_key(thrust::execution_policy<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init,
                                       BinaryPredicate binary_pred,
                                       AssociativeOperator binary_op)
{
  typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;
  typedef unsigned int HeadFlagType;

  const size_t n = last1 - first1;

  if(n != 0)
  {
    // compute head flags
    typedef detail::swap_arguments_binary_predicate<BinaryPredicate> SwappedBinaryPredicate;
    typedef thrust::detail::head_flags<InputIterator1, SwappedBinaryPredicate> HeadFlags;
    HeadFlags head_flags(first1, last1, SwappedBinaryPredicate(binary_pred));

    typedef thrust::counting_iterator<std::size_t> CountingIterator;
    typedef thrust::tuple<CountingIterator, typename HeadFlags::iterator> IteratorTuple;
    typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
    
    ZipIterator zip_it(thrust::make_tuple(CountingIterator(0), head_flags.begin()));

    typedef init_functor<InputIterator2, T> InitFunction;
    typedef thrust::transform_iterator<InitFunction, ZipIterator> InitTransformIterator;

    InitTransformIterator transform_it(zip_it, InitFunction(first2, init));

    // scan key-flag tuples,
    // For additional details refer to Section 2 of the following paper
    //    S. Sengupta, M. Harris, and M. Garland. "Efficient parallel scan algorithms for GPUs"
    //    NVIDIA Technical Report NVR-2008-003, December 2008
    //    http://mgarland.org/files/papers/nvr-2008-003.pdf
    
    
    // in-place
    if (iterator_equal(first2, result))
    {
        thrust::detail::temporary_array<OutputType, DerivedPolicy> temp(exec, n);
        thrust::inclusive_scan(exec,
                               thrust::make_zip_iterator(thrust::make_tuple(transform_it, head_flags.begin())),
                               thrust::make_zip_iterator(thrust::make_tuple(transform_it, head_flags.begin())) + n,
                               thrust::make_zip_iterator(thrust::make_tuple(temp.begin(), thrust::make_discard_iterator())),
                               detail::segmented_scan_functor<OutputType, HeadFlagType, AssociativeOperator>(binary_op));
        thrust::copy(exec, temp.begin(), temp.end(), result);
    }
    else
    {
        thrust::inclusive_scan(exec,
                               thrust::make_zip_iterator(thrust::make_tuple(transform_it, head_flags.begin())),
                               thrust::make_zip_iterator(thrust::make_tuple(transform_it, head_flags.begin())) + n,
                               thrust::make_zip_iterator(thrust::make_tuple(result,       thrust::make_discard_iterator())),
                               detail::segmented_scan_functor<OutputType, HeadFlagType, AssociativeOperator>(binary_op));
    }
  }

  return result + n;
}


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace thrust

