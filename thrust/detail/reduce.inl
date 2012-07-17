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


/*! \file reduce.inl
 *  \brief Inline file for reduce.h.
 */

#include <thrust/reduce.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/reduce.h>
#include <thrust/system/detail/generic/reduce_by_key.h>
#include <thrust/system/detail/adl/reduce.h>
#include <thrust/system/detail/adl/reduce_by_key.h>

namespace thrust
{


template<typename System, typename InputIterator>
  typename thrust::iterator_traits<InputIterator>::value_type
    reduce(thrust::detail::dispatchable_base<System> &system, InputIterator first, InputIterator last)
{
  using thrust::system::detail::generic::reduce;
  return reduce(system.derived(), first, last);
} // end reduce()


template<typename System, typename InputIterator, typename T>
  T reduce(thrust::detail::dispatchable_base<System> &system,
           InputIterator first,
           InputIterator last,
           T init)
{
  using thrust::system::detail::generic::reduce;
  return reduce(system.derived(), first, last, init);
} // end reduce()


template<typename System,
         typename InputIterator,
         typename T,
         typename BinaryFunction>
  T reduce(thrust::detail::dispatchable_base<System> &system,
           InputIterator first,
           InputIterator last,
           T init,
           BinaryFunction binary_op)
{
  using thrust::system::detail::generic::reduce;
  return reduce(system.derived(), first, last, init, binary_op);
} // end reduce()


template <typename System,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(thrust::detail::dispatchable_base<System> &system,
                InputIterator1 keys_first, 
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output)
{
  using thrust::system::detail::generic::reduce_by_key;
  return reduce_by_key(system.derived(), keys_first, keys_last, values_first, keys_output, values_output);
} // end reduce_by_key()


template <typename System,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(thrust::detail::dispatchable_base<System> &system,
                InputIterator1 keys_first, 
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred)
{
  using thrust::system::detail::generic::reduce_by_key;
  return reduce_by_key(system.derived(), keys_first, keys_last, values_first, keys_output, values_output, binary_pred);
} // end reduce_by_key()


template <typename System,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate,
          typename BinaryFunction>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(thrust::detail::dispatchable_base<System> &system,
                InputIterator1 keys_first, 
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred,
                BinaryFunction binary_op)
{
  using thrust::system::detail::generic::reduce_by_key;
  return reduce_by_key(system.derived(), keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);
} // end reduce_by_key()


namespace detail
{


template<typename System, typename InputIterator>
  typename thrust::iterator_traits<InputIterator>::value_type
    strip_const_reduce(const System &system, InputIterator first, InputIterator last)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::reduce(non_const_system, first, last);
} // end reduce()


template<typename System, typename InputIterator, typename T>
  T strip_const_reduce(const System &system,
                       InputIterator first,
                       InputIterator last,
                       T init)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::reduce(non_const_system, first, last, init);
} // end reduce()


template<typename System,
         typename InputIterator,
         typename T,
         typename BinaryFunction>
  T strip_const_reduce(const System &system,
                       InputIterator first,
                       InputIterator last,
                       T init,
                       BinaryFunction binary_op)
{
  System &non_const_system = const_cast<System&>(system);
  return thrust::reduce(non_const_system, first, last, init, binary_op);
} // end reduce()


template <typename System,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
    strip_const_reduce_by_key(const System &system,
                              InputIterator1 keys_first, 
                              InputIterator1 keys_last,
                              InputIterator2 values_first,
                              OutputIterator1 keys_output,
                              OutputIterator2 values_output)
{
  System &non_const_cast = const_cast<System&>(system);
  return thrust::reduce_by_key(non_const_cast, keys_first, keys_last, values_first, keys_output, values_output);
} // end reduce_by_key()


template <typename System,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate>
  thrust::pair<OutputIterator1,OutputIterator2>
    strip_const_reduce_by_key(const System &system,
                              InputIterator1 keys_first, 
                              InputIterator1 keys_last,
                              InputIterator2 values_first,
                              OutputIterator1 keys_output,
                              OutputIterator2 values_output,
                              BinaryPredicate binary_pred)
{
  System &non_const_cast = const_cast<System&>(system);
  return thrust::reduce_by_key(non_const_cast, keys_first, keys_last, values_first, keys_output, values_output, binary_pred);
} // end reduce_by_key()


template <typename System,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate,
          typename BinaryFunction>
  thrust::pair<OutputIterator1,OutputIterator2>
    strip_const_reduce_by_key(const System &system,
                              InputIterator1 keys_first, 
                              InputIterator1 keys_last,
                              InputIterator2 values_first,
                              OutputIterator1 keys_output,
                              OutputIterator2 values_output,
                              BinaryPredicate binary_pred,
                              BinaryFunction binary_op)
{
  System &non_const_cast = const_cast<System&>(system);
  return thrust::reduce_by_key(non_const_cast, keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);
} // end reduce_by_key()


} // end detail


template<typename InputIterator>
typename thrust::iterator_traits<InputIterator>::value_type
  reduce(InputIterator first,
         InputIterator last)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type system;

  return thrust::detail::strip_const_reduce(select_system(system()), first, last);
}

template<typename InputIterator,
         typename T>
   T reduce(InputIterator first,
            InputIterator last,
            T init)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type system;

  return thrust::detail::strip_const_reduce(select_system(system()), first, last, init);
}


template<typename InputIterator,
         typename T,
         typename BinaryFunction>
   T reduce(InputIterator first,
            InputIterator last,
            T init,
            BinaryFunction binary_op)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type system;

  return thrust::detail::strip_const_reduce(select_system(system()), first, last, init, binary_op);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(InputIterator1 keys_first, 
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type  system1;
  typedef typename thrust::iterator_system<InputIterator2>::type  system2;
  typedef typename thrust::iterator_system<OutputIterator1>::type system3;
  typedef typename thrust::iterator_system<OutputIterator2>::type system4;

  return thrust::detail::strip_const_reduce_by_key(select_system(system1(),system2(),system3(),system4()), keys_first, keys_last, values_first, keys_output, values_output);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(InputIterator1 keys_first, 
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type  system1;
  typedef typename thrust::iterator_system<InputIterator2>::type  system2;
  typedef typename thrust::iterator_system<OutputIterator1>::type system3;
  typedef typename thrust::iterator_system<OutputIterator2>::type system4;

  return thrust::detail::strip_const_reduce_by_key(select_system(system1(),system2(),system3(),system4()), keys_first, keys_last, values_first, keys_output, values_output, binary_pred);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate,
          typename BinaryFunction>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(InputIterator1 keys_first, 
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred,
                BinaryFunction binary_op)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type  system1;
  typedef typename thrust::iterator_system<InputIterator2>::type  system2;
  typedef typename thrust::iterator_system<OutputIterator1>::type system3;
  typedef typename thrust::iterator_system<OutputIterator2>::type system4;

  return thrust::detail::strip_const_reduce_by_key(select_system(system1(),system2(),system3(),system4()), keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);
}

} // end namespace thrust

