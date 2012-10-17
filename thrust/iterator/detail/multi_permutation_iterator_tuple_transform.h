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

#pragma once

//
// this code is a workaround for a bug in nvcc
// see multi_permutation_iterator_base.h for an explanation
//
// note: we do not re-implement do_it_on_the_host / tuple_host_transform
//       since it is no longer needed
//
#ifdef WAR_NVCC_CANNOT_HANDLE_DEPENDENT_TEMPLATE_TEMPLATE_ARGUMENT

#include <thrust/tuple.h>

namespace thrust
{

namespace detail
{

namespace multi_permutation_iterator_tuple_transform_ns
{

template<typename Tuple,
         typename XfrmTuple,
         typename UnaryFunction,
         unsigned int sz = thrust::tuple_size<Tuple>::value>
  struct tuple_transform_functor;


template<typename Tuple,
         typename XfrmTuple,
         typename UnaryFunction>
  struct tuple_transform_functor<Tuple,XfrmTuple,UnaryFunction,0>
{
  static __host__ __device__
  XfrmTuple
  do_it_on_the_host_or_device(const Tuple &t, UnaryFunction f)
  {    return thrust::null_type();
  }
};


template<typename Tuple,
         typename XfrmTuple,
         typename UnaryFunction>
  struct tuple_transform_functor<Tuple,XfrmTuple,UnaryFunction,1>
{
  static __host__ __device__
  XfrmTuple
  do_it_on_the_host_or_device(const Tuple &t, UnaryFunction f)
  {
    return XfrmTuple(f(thrust::get<0>(t)));
  }
};


template<typename Tuple,
         typename XfrmTuple,
         typename UnaryFunction>
  struct tuple_transform_functor<Tuple,XfrmTuple,UnaryFunction,2>
{
  static __host__ __device__
  XfrmTuple
  do_it_on_the_host_or_device(const Tuple &t, UnaryFunction f)
  {
    return XfrmTuple(f(thrust::get<0>(t)),
                     f(thrust::get<1>(t)));
  }
};


template<typename Tuple,
         typename XfrmTuple,
         typename UnaryFunction>
  struct tuple_transform_functor<Tuple,XfrmTuple,UnaryFunction,3>
{
  static __host__ __device__
  XfrmTuple
  do_it_on_the_host_or_device(const Tuple &t, UnaryFunction f)
  {
    return XfrmTuple(f(thrust::get<0>(t)),
                     f(thrust::get<1>(t)),
                     f(thrust::get<2>(t)));
  }
};


template<typename Tuple,
         typename XfrmTuple,
         typename UnaryFunction>
  struct tuple_transform_functor<Tuple,XfrmTuple,UnaryFunction,4>
{
  static __host__ __device__
  XfrmTuple
  do_it_on_the_host_or_device(const Tuple &t, UnaryFunction f)
  {
    return XfrmTuple(f(thrust::get<0>(t)),
                     f(thrust::get<1>(t)),
                     f(thrust::get<2>(t)),
                     f(thrust::get<3>(t)));
  }
};


template<typename Tuple,
         typename XfrmTuple,
         typename UnaryFunction>
  struct tuple_transform_functor<Tuple,XfrmTuple,UnaryFunction,5>
{
  static __host__ __device__
  XfrmTuple
  do_it_on_the_host_or_device(const Tuple &t, UnaryFunction f)
  {
    return XfrmTuple(f(thrust::get<0>(t)),
                     f(thrust::get<1>(t)),
                     f(thrust::get<2>(t)),
                     f(thrust::get<3>(t)),
                     f(thrust::get<4>(t)));
  }
};


template<typename Tuple,
         typename XfrmTuple,
         typename UnaryFunction>
  struct tuple_transform_functor<Tuple,XfrmTuple,UnaryFunction,6>
{
  static __host__ __device__
  XfrmTuple
  do_it_on_the_host_or_device(const Tuple &t, UnaryFunction f)
  {
    return XfrmTuple(f(thrust::get<0>(t)),
                     f(thrust::get<1>(t)),
                     f(thrust::get<2>(t)),
                     f(thrust::get<3>(t)),
                     f(thrust::get<4>(t)),
                     f(thrust::get<5>(t)));
  }
};


template<typename Tuple,
         typename XfrmTuple,
         typename UnaryFunction>
  struct tuple_transform_functor<Tuple,XfrmTuple,UnaryFunction,7>
{
  static __host__ __device__
  XfrmTuple
  do_it_on_the_host_or_device(const Tuple &t, UnaryFunction f)
  {
    return XfrmTuple(f(thrust::get<0>(t)),
                     f(thrust::get<1>(t)),
                     f(thrust::get<2>(t)),
                     f(thrust::get<3>(t)),
                     f(thrust::get<4>(t)),
                     f(thrust::get<5>(t)),
                     f(thrust::get<6>(t)));
  }
};


template<typename Tuple,
         typename XfrmTuple,
         typename UnaryFunction>
  struct tuple_transform_functor<Tuple,XfrmTuple,UnaryFunction,8>
{
  static __host__ __device__
  XfrmTuple
  do_it_on_the_host_or_device(const Tuple &t, UnaryFunction f)
  {
    return XfrmTuple(f(thrust::get<0>(t)),
                     f(thrust::get<1>(t)),
                     f(thrust::get<2>(t)),
                     f(thrust::get<3>(t)),
                     f(thrust::get<4>(t)),
                     f(thrust::get<5>(t)),
                     f(thrust::get<6>(t)),
                     f(thrust::get<7>(t)));
  }
};


template<typename Tuple,
         typename XfrmTuple,
         typename UnaryFunction>
  struct tuple_transform_functor<Tuple,XfrmTuple,UnaryFunction,9>
{
  static __host__ __device__
  XfrmTuple
  do_it_on_the_host_or_device(const Tuple &t, UnaryFunction f)
  {
    return XfrmTuple(f(thrust::get<0>(t)),
                     f(thrust::get<1>(t)),
                     f(thrust::get<2>(t)),
                     f(thrust::get<3>(t)),
                     f(thrust::get<4>(t)),
                     f(thrust::get<5>(t)),
                     f(thrust::get<6>(t)),
                     f(thrust::get<7>(t)),
                     f(thrust::get<8>(t)));
  }
};


template<typename Tuple,
         typename XfrmTuple,
         typename UnaryFunction>
  struct tuple_transform_functor<Tuple,XfrmTuple,UnaryFunction,10>
{
  static __host__ __device__
  XfrmTuple
  do_it_on_the_host_or_device(const Tuple &t, UnaryFunction f)
  {
    return XfrmTuple(f(thrust::get<0>(t)),
                     f(thrust::get<1>(t)),
                     f(thrust::get<2>(t)),
                     f(thrust::get<3>(t)),
                     f(thrust::get<4>(t)),
                     f(thrust::get<5>(t)),
                     f(thrust::get<6>(t)),
                     f(thrust::get<7>(t)),
                     f(thrust::get<8>(t)),
                     f(thrust::get<9>(t)));
  }
};

template<typename XfrmTuple,
         typename Tuple,
         typename UnaryFunction>
XfrmTuple
__host__ __device__
tuple_host_device_transform(const Tuple &t, UnaryFunction f)
{
  return tuple_transform_functor<Tuple,XfrmTuple,UnaryFunction>::do_it_on_the_host_or_device(t,f);
}

} // end multi_permutation_iterator_tuple_transform_ns

} // end detail

} // end thrust

#endif

