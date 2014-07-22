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
#include <thrust/reduce.h>
#include <thrust/detail/seq.h>
#include <thrust/system/cuda/detail/reduce_by_key.h>
#include <thrust/system/cuda/detail/bulk.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/detail/range/head_flags.h>
#include <thrust/detail/range/tail_flags.h>
#include <thrust/system/cuda/detail/reduce_intervals.hpp>


namespace thrust
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace reduce_by_key_detail
{


struct reduce_by_key_kernel
{
  template<typename ConcurrentGroup,
           typename RandomAccessIterator1,
           typename Decomposition,
           typename RandomAccessIterator2,
           typename RandomAccessIterator3,
           typename RandomAccessIterator4,
           typename RandomAccessIterator5,
           typename RandomAccessIterator6,
           typename RandomAccessIterator7,
           typename BinaryPredicate,
           typename BinaryFunction>
  __device__
  thrust::pair<RandomAccessIterator3,RandomAccessIterator4>
  operator()(ConcurrentGroup &g,
             RandomAccessIterator1 keys_first,
             Decomposition decomp,
             RandomAccessIterator2 values_first,
             RandomAccessIterator3 keys_result,
             RandomAccessIterator4 values_result,
             RandomAccessIterator5 interval_output_offsets,
             RandomAccessIterator6 interval_values,
             RandomAccessIterator7 is_carry,
             //BinaryPredicate pred,
             //BinaryFunction binary_op)
             thrust::tuple<BinaryPredicate,BinaryFunction> pred_and_binary_op)
  {
    typedef typename thrust::iterator_value<RandomAccessIterator1>::type key_type;
    typedef typename thrust::iterator_value<RandomAccessIterator2>::type value_type;

    BinaryPredicate pred = thrust::get<0>(pred_and_binary_op);
    BinaryFunction binary_op = thrust::get<1>(pred_and_binary_op);

    thrust::detail::tail_flags<RandomAccessIterator1,BinaryPredicate> tail_flags(keys_first, keys_first + decomp.n(), pred);

    typename Decomposition::size_type input_first, input_last;
    thrust::tie(input_first,input_last) = decomp[g.index()];

    typename Decomposition::size_type output_first = g.index() == 0 ? 0 : interval_output_offsets[g.index() - 1];

    key_type init_key     = keys_first[input_first];
    value_type init_value = values_first[input_first];

    // the inits become the carries
    thrust::tie(keys_result, values_result, init_key, init_value) =
      bulk_::reduce_by_key(g,
                           keys_first + input_first + 1,
                           keys_first + input_last,
                           values_first + input_first + 1,
                           keys_result + output_first,
                           values_result + output_first,
                           init_key,
                           init_value,
                           pred,
                           binary_op);

    if(g.this_exec.index() == 0)
    {
      bool interval_has_carry = !tail_flags[input_last-1];

      if(interval_has_carry)
      {
        interval_values[g.index()] = init_value;
      } // end if
      else
      {
        *keys_result   = init_key;
        *values_result = init_value;

        ++keys_result;
        ++values_result;
      } // end else

      is_carry[g.index()] = interval_has_carry;
    } // end if

    return thrust::make_pair(keys_result, values_result);
  }


  template<typename ConcurrentGroup,
           typename RandomAccessIterator1,
           typename RandomAccessIterator2,
           typename RandomAccessIterator3,
           typename RandomAccessIterator4,
           typename BinaryPredicate,
           typename BinaryFunction,
           typename Iterator>
  __device__
  void operator()(ConcurrentGroup      &g,
                  RandomAccessIterator1 keys_first,
                  RandomAccessIterator1 keys_last,
                  RandomAccessIterator2 values_first,
                  RandomAccessIterator3 keys_result,
                  RandomAccessIterator4 values_result,
                  BinaryPredicate       pred,
                  BinaryFunction        binary_op,
                  Iterator result_size)
  {
    RandomAccessIterator3 old_keys_result = keys_result;

    thrust::tie(keys_result, values_result) =
      operator()(g, keys_first, make_trivial_decomposition(keys_last - keys_first), values_first, keys_result, values_result,
                 thrust::make_constant_iterator<int>(0),
                 thrust::make_discard_iterator(),
                 thrust::make_discard_iterator(),
                 thrust::make_tuple(pred,binary_op));

    if(g.this_exec.index() == 0)
    {
      *result_size = keys_result - old_keys_result;
    }
  }
};


struct tuple_and
{
  typedef bool result_type;

  template<typename Tuple>
  __host__ __device__
  bool operator()(Tuple t)
  {
    return thrust::get<0>(t) && thrust::get<1>(t);
  }
};


template<typename DerivedPolicy,
         typename Iterator1,
         typename Iterator2,
         typename Iterator3,
         typename Iterator4,
         typename BinaryFunction>
__host__ __device__
void sum_tail_carries(execution_policy<DerivedPolicy> &exec,
                      Iterator1 interval_values_first,
                      Iterator1 interval_values_last,
                      Iterator2 interval_output_offsets_first,
                      Iterator2 interval_output_offsets_last,
                      Iterator3 is_carry,
                      Iterator4 values_result,
                      BinaryFunction binary_op)
{
  typedef thrust::zip_iterator<thrust::tuple<Iterator2,Iterator3> > zip_iterator;

  thrust::detail::tail_flags<zip_iterator> tail_flags(thrust::make_zip_iterator(thrust::make_tuple(interval_output_offsets_first, is_carry)),
                                                      thrust::make_zip_iterator(thrust::make_tuple(interval_output_offsets_last,  is_carry)));

  // for each value in the array of interval values
  //   if it is a carry and it is the tail value in its segment
  //     scatter it to its location in the output array, but sum it together with the value there previously
  thrust::transform_if(exec,
                       interval_values_first, interval_values_last,
                       thrust::make_permutation_iterator(values_result, interval_output_offsets_first),
                       thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(tail_flags.begin(), is_carry)), tuple_and()),
                       thrust::make_permutation_iterator(values_result, interval_output_offsets_first),
                       binary_op,
                       thrust::identity<bool>());
} // end sum_tail_carries()


template<typename InputIterator, typename OutputIterator, typename BinaryFunction>
struct intermediate_type
  : thrust::detail::eval_if<
    thrust::detail::has_result_type<BinaryFunction>::value,
    thrust::detail::result_type<BinaryFunction>,
    thrust::detail::eval_if<
      thrust::detail::is_output_iterator<OutputIterator>::value,
      thrust::iterator_value<InputIterator>,
      thrust::iterator_value<OutputIterator>
    >
  >
{};


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction>
__host__ __device__
thrust::pair<OutputIterator1,OutputIterator2>
reduce_by_key(execution_policy<DerivedPolicy> &exec,
              InputIterator1 keys_first, 
              InputIterator1 keys_last,
              InputIterator2 values_first,
              OutputIterator1 keys_result,
              OutputIterator2 values_result,
              BinaryPredicate binary_pred,
              BinaryFunction binary_op)
{
  typedef typename thrust::iterator_difference<InputIterator1>::type difference_type;
  typedef typename thrust::iterator_value<InputIterator2>::type      value_type;
  typedef int size_type;

  const difference_type n = keys_last - keys_first;

  if(n <= 0) return thrust::make_pair(keys_result, values_result);

  const size_type threshold_of_parallelism = 20000;

  if(n <= threshold_of_parallelism)
  {
    thrust::detail::temporary_array<size_type,DerivedPolicy> result_size_storage(exec, 1);

    // XXX these sizes aren't actually optimal, but anything larger
    //     will cause sm_1x to run out of smem at compile time
    // XXX all of this grossness would go away if we could rely on shmalloc
    const int groupsize =
      (sizeof(value_type) <=     sizeof(int)) ? 512 :
      (sizeof(value_type) <= 2 * sizeof(int)) ? 256 :
      128;

    const int grainsize = (sizeof(value_type) == sizeof(int)) ? 3 : 5;

    size_type heap_size = groupsize * grainsize * (sizeof(size_type) + sizeof(value_type));
    bulk_::async(bulk_::grid<groupsize,grainsize>(1,heap_size,stream(thrust::detail::derived_cast(exec))), reduce_by_key_detail::reduce_by_key_kernel(),
      bulk_::root.this_exec, keys_first, keys_last, values_first, keys_result, values_result, binary_pred, binary_op, result_size_storage.begin());

    size_type result_size = result_size_storage[0];

    return thrust::make_pair(keys_result + result_size, values_result + result_size);
  } // end if

  typedef typename reduce_by_key_detail::intermediate_type<
    InputIterator2, OutputIterator2, BinaryFunction
  >::type intermediate_type;

  const size_type groupsize = 128;
  const size_type grainsize = 5;
  size_type tile_size = groupsize * grainsize;

  const size_type interval_size = threshold_of_parallelism; 

  size_type subscription = 100;
  size_type num_groups = thrust::min<size_type>(subscription * bulk_::concurrent_group<>::hardware_concurrency(), (n + interval_size - 1) / interval_size);
  aligned_decomposition<size_type> decomp(n, num_groups, tile_size);

  // count the number of tail flags in each interval
  thrust::detail::tail_flags<
    InputIterator1,
    BinaryPredicate,
    size_type
  > tail_flags(keys_first, keys_last, binary_pred);

  thrust::detail::temporary_array<size_type,DerivedPolicy> interval_output_offsets(exec, decomp.size());

  reduce_intervals_(exec, tail_flags.begin(), decomp, interval_output_offsets.begin(), thrust::plus<size_type>());

  // scan the interval counts
  thrust::inclusive_scan(exec, interval_output_offsets.begin(), interval_output_offsets.end(), interval_output_offsets.begin());

  // reduce each interval
  thrust::detail::temporary_array<bool,DerivedPolicy> is_carry(exec, decomp.size());
  thrust::detail::temporary_array<intermediate_type,DerivedPolicy> interval_values(exec, decomp.size());

  size_type heap_size = tile_size * (sizeof(size_type) + sizeof(value_type));
  bulk_::async(bulk_::grid<groupsize,grainsize>(decomp.size(),heap_size,stream(thrust::detail::derived_cast(exec))), reduce_by_key_detail::reduce_by_key_kernel(),
    bulk_::root.this_exec, keys_first, decomp, values_first, keys_result, values_result, interval_output_offsets.begin(), interval_values.begin(), is_carry.begin(), thrust::make_tuple(binary_pred, binary_op)
  );

  // scan by key the carries
  thrust::inclusive_scan_by_key(exec,
                                thrust::make_zip_iterator(thrust::make_tuple(interval_output_offsets.begin(), is_carry.begin())),
                                thrust::make_zip_iterator(thrust::make_tuple(interval_output_offsets.end(),   is_carry.end())),
                                interval_values.begin(),
                                interval_values.begin(),
                                thrust::equal_to<thrust::tuple<size_type,bool> >(),
                                binary_op);

  // sum each tail carry value into the result 
  reduce_by_key_detail::sum_tail_carries(exec,
                                         interval_values.begin(), interval_values.end(),
                                         interval_output_offsets.begin(), interval_output_offsets.end(),
                                         is_carry.begin(),
                                         values_result,
                                         binary_op);

  difference_type result_size = interval_output_offsets[interval_output_offsets.size() - 1];

  return thrust::make_pair(keys_result + result_size, values_result + result_size);
} // end reduce_by_key()


} // end namespace reduce_by_key_detail


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction>
__host__ __device__
thrust::pair<OutputIterator1,OutputIterator2>
reduce_by_key(execution_policy<DerivedPolicy> &exec,
              InputIterator1 keys_first, 
              InputIterator1 keys_last,
              InputIterator2 values_first,
              OutputIterator1 keys_result,
              OutputIterator2 values_result,
              BinaryPredicate binary_pred,
              BinaryFunction binary_op)
{
  // we're attempting to launch a kernel, assert we're compiling with nvcc
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to compile your code using nvcc, rather than g++ or cl.exe  X
  // ========================================================================
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<InputIterator1, THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC>::value) );

  struct workaround
  {
    static __host__ __device__
    thrust::pair<OutputIterator1,OutputIterator2>
    parallel_path(execution_policy<DerivedPolicy> &exec,
                  InputIterator1 keys_first,
                  InputIterator1 keys_last,
                  InputIterator2 values_first,
                  OutputIterator1 keys_result,
                  OutputIterator2 values_result,
                  BinaryPredicate binary_pred,
                  BinaryFunction binary_op)
    {
      return thrust::system::cuda::detail::reduce_by_key_detail::reduce_by_key(exec, keys_first, keys_last, values_first, keys_result, values_result, binary_pred, binary_op);
    }

    static __host__ __device__
    thrust::pair<OutputIterator1,OutputIterator2>
    sequential_path(execution_policy<DerivedPolicy> &,
                    InputIterator1 keys_first,
                    InputIterator1 keys_last,
                    InputIterator2 values_first,
                    OutputIterator1 keys_result,
                    OutputIterator2 values_result,
                    BinaryPredicate binary_pred,
                    BinaryFunction binary_op)
    {
      return thrust::reduce_by_key(thrust::seq, keys_first, keys_last, values_first, keys_result, values_result, binary_pred, binary_op);
    }
  };

#if __BULK_HAS_CUDART__
  return workaround::parallel_path(exec, keys_first, keys_last, values_first, keys_result, values_result, binary_pred, binary_op);
#else
  return workaround::sequential_path(exec, keys_first, keys_last, values_first, keys_result, values_result, binary_pred, binary_op);
#endif
} // end reduce_by_key()


} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace thrust

