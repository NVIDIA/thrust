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

#pragma once

#include <thrust/system/cuda/detail/bulk/detail/config.hpp>
#include <thrust/system/cuda/detail/bulk/detail/tuple_meta_transform.hpp>
#include <thrust/tuple.h>

BULK_NAMESPACE_PREFIX
namespace bulk
{
namespace detail
{

template<typename Tuple,
         template<typename> class UnaryMetaFunction,
         typename UnaryFunction,
         unsigned int sz = thrust::tuple_size<Tuple>::value>
  struct tuple_transform_functor;


template<typename Tuple,
         template<typename> class UnaryMetaFunction,
         typename UnaryFunction>
  struct tuple_transform_functor<Tuple,UnaryMetaFunction,UnaryFunction,0>
{
  static __host__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it_on_the_host(const Tuple &t, UnaryFunction f)
  {
    return thrust::tuple<>();
  }

  static __host__ __device__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it_on_the_host_or_device(const Tuple &t, UnaryFunction f)
  {
    return thrust::tuple<>();
  }
};


template<typename Tuple,
         template<typename> class UnaryMetaFunction,
         typename UnaryFunction>
  struct tuple_transform_functor<Tuple,UnaryMetaFunction,UnaryFunction,1>
{
  static __host__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it_on_the_host(const Tuple &t, UnaryFunction f)
  {
    typedef typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type XfrmTuple;

    return XfrmTuple(f(thrust::get<0>(t)));
  }

  static __host__ __device__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it_on_the_host_or_device(const Tuple &t, UnaryFunction f)
  {
    typedef typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type XfrmTuple;

    return XfrmTuple(f(thrust::get<0>(t)));
  }
};


template<typename Tuple,
         template<typename> class UnaryMetaFunction,
         typename UnaryFunction>
  struct tuple_transform_functor<Tuple,UnaryMetaFunction,UnaryFunction,2>
{
  static __host__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it_on_the_host(const Tuple &t, UnaryFunction f)
  {
    typedef typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type XfrmTuple;

    return XfrmTuple(f(thrust::get<0>(t)),
                     f(thrust::get<1>(t)));
  }

  static __host__ __device__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it_on_the_host_or_device(const Tuple &t, UnaryFunction f)
  {
    typedef typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type XfrmTuple;

    return XfrmTuple(f(thrust::get<0>(t)),
                     f(thrust::get<1>(t)));
  }
};


template<typename Tuple,
         template<typename> class UnaryMetaFunction,
         typename UnaryFunction>
  struct tuple_transform_functor<Tuple,UnaryMetaFunction,UnaryFunction,3>
{
  static __host__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it_on_the_host(const Tuple &t, UnaryFunction f)
  {
    typedef typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type XfrmTuple;

    return XfrmTuple(f(thrust::get<0>(t)),
                     f(thrust::get<1>(t)),
                     f(thrust::get<2>(t)));
  }

  static __host__ __device__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it_on_the_host_or_device(const Tuple &t, UnaryFunction f)
  {
    typedef typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type XfrmTuple;

    return XfrmTuple(f(thrust::get<0>(t)),
                     f(thrust::get<1>(t)),
                     f(thrust::get<2>(t)));
  }
};


template<typename Tuple,
         template<typename> class UnaryMetaFunction,
         typename UnaryFunction>
  struct tuple_transform_functor<Tuple,UnaryMetaFunction,UnaryFunction,4>
{
  static __host__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it_on_the_host(const Tuple &t, UnaryFunction f)
  {
    typedef typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type XfrmTuple;

    return XfrmTuple(f(thrust::get<0>(t)),
                     f(thrust::get<1>(t)),
                     f(thrust::get<2>(t)),
                     f(thrust::get<3>(t)));
  }

  static __host__ __device__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it_on_the_host_or_device(const Tuple &t, UnaryFunction f)
  {
    typedef typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type XfrmTuple;

    return XfrmTuple(f(thrust::get<0>(t)),
                     f(thrust::get<1>(t)),
                     f(thrust::get<2>(t)),
                     f(thrust::get<3>(t)));
  }
};


template<typename Tuple,
         template<typename> class UnaryMetaFunction,
         typename UnaryFunction>
  struct tuple_transform_functor<Tuple,UnaryMetaFunction,UnaryFunction,5>
{
  static __host__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it_on_the_host(const Tuple &t, UnaryFunction f)
  {
    typedef typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type XfrmTuple;

    return XfrmTuple(f(thrust::get<0>(t)),
                     f(thrust::get<1>(t)),
                     f(thrust::get<2>(t)),
                     f(thrust::get<3>(t)),
                     f(thrust::get<4>(t)));
  }

  static __host__ __device__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it_on_the_host_or_device(const Tuple &t, UnaryFunction f)
  {
    typedef typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type XfrmTuple;

    return XfrmTuple(f(thrust::get<0>(t)),
                     f(thrust::get<1>(t)),
                     f(thrust::get<2>(t)),
                     f(thrust::get<3>(t)),
                     f(thrust::get<4>(t)));
  }
};


template<typename Tuple,
         template<typename> class UnaryMetaFunction,
         typename UnaryFunction>
  struct tuple_transform_functor<Tuple,UnaryMetaFunction,UnaryFunction,6>
{
  static __host__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it_on_the_host(const Tuple &t, UnaryFunction f)
  {
    typedef typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type XfrmTuple;

    return XfrmTuple(f(thrust::get<0>(t)),
                     f(thrust::get<1>(t)),
                     f(thrust::get<2>(t)),
                     f(thrust::get<3>(t)),
                     f(thrust::get<4>(t)),
                     f(thrust::get<5>(t)));
  }

  static __host__ __device__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it_on_the_host_or_device(const Tuple &t, UnaryFunction f)
  {
    typedef typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type XfrmTuple;

    return XfrmTuple(f(thrust::get<0>(t)),
                     f(thrust::get<1>(t)),
                     f(thrust::get<2>(t)),
                     f(thrust::get<3>(t)),
                     f(thrust::get<4>(t)),
                     f(thrust::get<5>(t)));
  }
};


template<typename Tuple,
         template<typename> class UnaryMetaFunction,
         typename UnaryFunction>
  struct tuple_transform_functor<Tuple,UnaryMetaFunction,UnaryFunction,7>
{
  static __host__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it_on_the_host(const Tuple &t, UnaryFunction f)
  {
    typedef typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type XfrmTuple;

    return XfrmTuple(f(thrust::get<0>(t)),
                     f(thrust::get<1>(t)),
                     f(thrust::get<2>(t)),
                     f(thrust::get<3>(t)),
                     f(thrust::get<4>(t)),
                     f(thrust::get<5>(t)),
                     f(thrust::get<6>(t)));
  }

  static __host__ __device__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it_on_the_host_or_device(const Tuple &t, UnaryFunction f)
  {
    typedef typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type XfrmTuple;

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
         template<typename> class UnaryMetaFunction,
         typename UnaryFunction>
  struct tuple_transform_functor<Tuple,UnaryMetaFunction,UnaryFunction,8>
{
  static __host__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it_on_the_host(const Tuple &t, UnaryFunction f)
  {
    typedef typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type XfrmTuple;

    return XfrmTuple(f(thrust::get<0>(t)),
                     f(thrust::get<1>(t)),
                     f(thrust::get<2>(t)),
                     f(thrust::get<3>(t)),
                     f(thrust::get<4>(t)),
                     f(thrust::get<5>(t)),
                     f(thrust::get<6>(t)),
                     f(thrust::get<7>(t)));
  }

  static __host__ __device__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it_on_the_host_or_device(const Tuple &t, UnaryFunction f)
  {
    typedef typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type XfrmTuple;

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
         template<typename> class UnaryMetaFunction,
         typename UnaryFunction>
  struct tuple_transform_functor<Tuple,UnaryMetaFunction,UnaryFunction,9>
{
  static __host__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it_on_the_host(const Tuple &t, UnaryFunction f)
  {
    typedef typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type XfrmTuple;

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

  static __host__ __device__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it_on_the_host_or_device(const Tuple &t, UnaryFunction f)
  {
    typedef typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type XfrmTuple;

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
         template<typename> class UnaryMetaFunction,
         typename UnaryFunction>
  struct tuple_transform_functor<Tuple,UnaryMetaFunction,UnaryFunction,10>
{
  static __host__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it_on_the_host(const Tuple &t, UnaryFunction f)
  {
    typedef typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type XfrmTuple;

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

  static __host__ __device__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it_on_the_host_or_device(const Tuple &t, UnaryFunction f)
  {
    typedef typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type XfrmTuple;

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


template<template<typename> class UnaryMetaFunction,
         typename Tuple,
         typename UnaryFunction>
typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
tuple_host_transform(const Tuple &t, UnaryFunction f)
{
  return tuple_transform_functor<Tuple,UnaryMetaFunction,UnaryFunction>::do_it_on_the_host(t,f);
}

template<template<typename> class UnaryMetaFunction,
         typename Tuple,
         typename UnaryFunction>
typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
__host__ __device__
tuple_host_device_transform(const Tuple &t, UnaryFunction f)
{
  return tuple_transform_functor<Tuple,UnaryMetaFunction,UnaryFunction>::do_it_on_the_host_or_device(t,f);
}

} // end detail
} // end thrust
BULK_NAMESPACE_SUFFIX

