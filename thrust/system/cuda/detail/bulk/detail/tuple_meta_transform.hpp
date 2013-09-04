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
#include <thrust/tuple.h>

BULK_NAMESPACE_PREFIX
namespace bulk
{
namespace detail
{


template<typename Tuple,
         template<typename> class UnaryMetaFunction,
         unsigned int sz = thrust::tuple_size<Tuple>::value>
  struct tuple_meta_transform;

template<typename Tuple,
         template<typename> class UnaryMetaFunction>
  struct tuple_meta_transform<Tuple,UnaryMetaFunction,0>
{
  typedef thrust::tuple<> type;
};

template<typename Tuple,
         template<typename> class UnaryMetaFunction>
  struct tuple_meta_transform<Tuple,UnaryMetaFunction,1>
{
  typedef thrust::tuple<
    typename UnaryMetaFunction<typename thrust::tuple_element<0,Tuple>::type>::type
  > type;
};

template<typename Tuple,
         template<typename> class UnaryMetaFunction>
  struct tuple_meta_transform<Tuple,UnaryMetaFunction,2>
{
  typedef thrust::tuple<
    typename UnaryMetaFunction<typename thrust::tuple_element<0,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<1,Tuple>::type>::type
  > type;
};

template<typename Tuple,
         template<typename> class UnaryMetaFunction>
  struct tuple_meta_transform<Tuple,UnaryMetaFunction,3>
{
  typedef thrust::tuple<
    typename UnaryMetaFunction<typename thrust::tuple_element<0,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<1,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<2,Tuple>::type>::type
  > type;
};

template<typename Tuple,
         template<typename> class UnaryMetaFunction>
  struct tuple_meta_transform<Tuple,UnaryMetaFunction,4>
{
  typedef thrust::tuple<
    typename UnaryMetaFunction<typename thrust::tuple_element<0,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<1,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<2,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<3,Tuple>::type>::type
  > type;
};

template<typename Tuple,
         template<typename> class UnaryMetaFunction>
  struct tuple_meta_transform<Tuple,UnaryMetaFunction,5>
{
  typedef thrust::tuple<
    typename UnaryMetaFunction<typename thrust::tuple_element<0,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<1,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<2,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<3,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<4,Tuple>::type>::type
  > type;
};

template<typename Tuple,
         template<typename> class UnaryMetaFunction>
  struct tuple_meta_transform<Tuple,UnaryMetaFunction,6>
{
  typedef thrust::tuple<
    typename UnaryMetaFunction<typename thrust::tuple_element<0,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<1,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<2,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<3,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<4,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<5,Tuple>::type>::type
  > type;
};

template<typename Tuple,
         template<typename> class UnaryMetaFunction>
  struct tuple_meta_transform<Tuple,UnaryMetaFunction,7>
{
  typedef thrust::tuple<
    typename UnaryMetaFunction<typename thrust::tuple_element<0,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<1,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<2,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<3,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<4,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<5,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<6,Tuple>::type>::type
  > type;
};

template<typename Tuple,
         template<typename> class UnaryMetaFunction>
  struct tuple_meta_transform<Tuple,UnaryMetaFunction,8>
{
  typedef thrust::tuple<
    typename UnaryMetaFunction<typename thrust::tuple_element<0,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<1,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<2,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<3,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<4,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<5,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<6,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<7,Tuple>::type>::type
  > type;
};

template<typename Tuple,
         template<typename> class UnaryMetaFunction>
  struct tuple_meta_transform<Tuple,UnaryMetaFunction,9>
{
  typedef thrust::tuple<
    typename UnaryMetaFunction<typename thrust::tuple_element<0,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<1,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<2,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<3,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<4,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<5,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<6,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<7,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<8,Tuple>::type>::type
  > type;
};

template<typename Tuple,
         template<typename> class UnaryMetaFunction>
  struct tuple_meta_transform<Tuple,UnaryMetaFunction,10>
{
  typedef thrust::tuple<
    typename UnaryMetaFunction<typename thrust::tuple_element<0,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<1,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<2,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<3,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<4,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<5,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<6,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<7,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<8,Tuple>::type>::type,
    typename UnaryMetaFunction<typename thrust::tuple_element<9,Tuple>::type>::type
  > type;
};


} // end detail
} // end bulk
BULK_NAMESPACE_SUFFIX

