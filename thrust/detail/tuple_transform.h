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

#pragma once

#include <thrust/tuple.h>
#include <thrust/detail/tuple_meta_transform.h>

namespace thrust
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
  static __host__ __device__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it(const Tuple &t, UnaryFunction f)
  {
    return thrust::null_type();
  }
};


template<typename Tuple,
         template<typename> class UnaryMetaFunction,
         typename UnaryFunction>
  struct tuple_transform_functor<Tuple,UnaryMetaFunction,UnaryFunction,1>
{
  static __host__ __device__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it(const Tuple &t, UnaryFunction f)
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
  static __host__ __device__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it(const Tuple &t, UnaryFunction f)
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
  static __host__ __device__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it(const Tuple &t, UnaryFunction f)
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
  static __host__ __device__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it(const Tuple &t, UnaryFunction f)
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
  static __host__ __device__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it(const Tuple &t, UnaryFunction f)
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
  static __host__ __device__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it(const Tuple &t, UnaryFunction f)
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
  static __host__ __device__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it(const Tuple &t, UnaryFunction f)
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
  static __host__ __device__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it(const Tuple &t, UnaryFunction f)
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
  static __host__ __device__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it(const Tuple &t, UnaryFunction f)
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
  static __host__ __device__
  typename tuple_meta_transform<Tuple,UnaryMetaFunction>::type
  do_it(const Tuple &t, UnaryFunction f)
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


template<typename Tuple,
         typename UnaryFunction>
__host__ __device__
typename tuple_meta_transform<Tuple,UnaryFunction::template apply>::type
tuple_transform(const Tuple &t, UnaryFunction f)
{
  return tuple_transform_functor<Tuple,UnaryFunction::template apply,UnaryFunction>::do_it(t,f);
}

} // end detail

} // end thrust

