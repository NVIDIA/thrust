#pragma once

#include <thrust/detail/config.h>
#include <thrust/tuple.h>
#include <thrust/detail/type_traits.h>

namespace thrust
{
namespace detail
{
namespace tuple_detail
{


// get the type of the ith element of a Tuple, unless i >= tuple_size<Tuple>,
// in which case return T
template<int i, typename Tuple, typename T>
struct get_or_result
  : eval_if<
    i < tuple_size<Tuple>::value,
    tuple_element<i,Tuple>,
    identity_<T>
>
{};


// get the ith element of a Tuple, unless i >= tuple_size<Tuple>,
// in which case return x
template<int i, typename Tuple, typename T>
inline __host__ __device__
typename lazy_enable_if<
  i < tuple_size<Tuple>::value,
  tuple_element<i,Tuple>
>::type
get_or(const Tuple &t, const T &val)
{
  return thrust::get<i>(t);
}


template<int i, typename Tuple, typename T>
inline __host__ __device__
typename enable_if<
  i >= tuple_size<Tuple>::value,
  const T&
>::type
get_or(const Tuple &t, const T &val)
{
  return val;
}


template<int i, typename Tuple, typename T>
inline __host__ __device__
typename enable_if<
  i >= tuple_size<Tuple>::value,
  T&
>::type
get_or(const Tuple &t, T &val)
{
  return val;
}


// get the type of the ith element of Tuple, unless i >= limit, in which case return null_type
template<int i, int limit, typename Tuple>
  struct tuple_element_or_null
    : eval_if<
        i < limit,
        tuple_element<i,Tuple>,
        identity_<null_type>
      >
{};


// get the type of the result of tuple_tail()
template<typename Tuple>
struct tuple_tail_result
{
  static const int limit = tuple_size<Tuple>::value;

  typedef thrust::tuple<
    typename tuple_element_or_null<1,limit,Tuple>::type,
    typename tuple_element_or_null<2,limit,Tuple>::type,
    typename tuple_element_or_null<3,limit,Tuple>::type,
    typename tuple_element_or_null<4,limit,Tuple>::type,
    typename tuple_element_or_null<5,limit,Tuple>::type,
    typename tuple_element_or_null<6,limit,Tuple>::type,
    typename tuple_element_or_null<7,limit,Tuple>::type,
    typename tuple_element_or_null<8,limit,Tuple>::type,
    typename tuple_element_or_null<9,limit,Tuple>::type
  > type;
};


template<typename Tuple>
inline __host__ __device__
typename enable_if<
  (tuple_size<Tuple>::value > 1),
  typename tuple_tail_result<Tuple>::type
>::type
  tuple_tail(const Tuple &t)
{
  typedef typename tuple_tail_result<Tuple>::type result_type;

  return result_type(get_or<1>(t, null_type()),
                     get_or<2>(t, null_type()),
                     get_or<3>(t, null_type()),
                     get_or<4>(t, null_type()),
                     get_or<5>(t, null_type()),
                     get_or<6>(t, null_type()),
                     get_or<7>(t, null_type()),
                     get_or<8>(t, null_type()),
                     get_or<9>(t, null_type()));
}


template<typename Tuple>
inline __host__ __device__
typename enable_if<
  (tuple_size<Tuple>::value <= 1),
  typename tuple_tail_result<Tuple>::type
>::type
  tuple_tail(const Tuple &t)
{
  typedef typename tuple_tail_result<Tuple>::type result_type;

  // XXX we require this extra overload of tuple_tail()
  //     because there is no thrust::tuple constructor
  //     which takes a null_type in the first parameter slot

  return result_type();
}


template<bool b, typename True, typename False>
inline __host__ __device__
typename enable_if<b, const True&>::type
if_else(const True &t, const False &f)
{
  return t;
}


template<bool b, typename True, typename False>
inline __host__ __device__
typename enable_if<b, True&>::type
if_else(True &t, const False &f)
{
  return t;
}


template<bool b, typename True, typename False>
inline __host__ __device__
typename enable_if<!b, False&>::type
if_else(const True &t, False &f)
{
  return f;
}


template<bool b, typename True, typename False>
inline __host__ __device__
typename enable_if<!b, const False&>::type
if_else(const True &t, const False &f)
{
  return f;
}


template<typename T, typename Tuple>
struct tuple_append_result
{
  static const int append_slot = thrust::tuple_size<Tuple>::value;

  template<int i>
  struct null_unless_append_slot
    : eval_if<
        i == append_slot,
        identity_<T>,
        identity_<null_type>
      >
  {};

  // to produce an element of the tuple,
  // get the ith element of t, unless i is larger than
  // t's size
  // in that case, use x when i == x_slot,
  // otherwise use null_type

  typedef thrust::tuple<
    typename get_or_result<0,Tuple, typename null_unless_append_slot<0>::type>::type,
    typename get_or_result<1,Tuple, typename null_unless_append_slot<1>::type>::type,
    typename get_or_result<2,Tuple, typename null_unless_append_slot<2>::type>::type,
    typename get_or_result<3,Tuple, typename null_unless_append_slot<3>::type>::type,
    typename get_or_result<4,Tuple, typename null_unless_append_slot<4>::type>::type,
    typename get_or_result<5,Tuple, typename null_unless_append_slot<5>::type>::type,
    typename get_or_result<6,Tuple, typename null_unless_append_slot<6>::type>::type,
    typename get_or_result<7,Tuple, typename null_unless_append_slot<7>::type>::type,
    typename get_or_result<8,Tuple, typename null_unless_append_slot<8>::type>::type,
    typename get_or_result<9,Tuple, typename null_unless_append_slot<9>::type>::type,
  > type;
};


// append x to a tuple, producing a copy of t with x appended
template<typename T, typename Tuple>
inline __host__ __device__
typename enable_if<
  tuple_size<Tuple>::value < 10,
  typename tuple_append_result<T,Tuple>::type
>::type
tuple_append(const Tuple &t, const T &x)
{
  // the slot into which x will go
  const int x_slot = thrust::tuple_size<Tuple>::value;

  typedef typename tuple_append_result<T,Tuple>::type result_type;

  // to produce an element of the tuple,
  // get the ith element of t, unless i is larger than
  // t's size
  // in that case, use x when i == x_slot,
  // otherwise use null_type
  
  const null_type null;

  return result_type(get_or<0>(t,if_else<0 == x_slot>(x, null)),
                     get_or<1>(t,if_else<1 == x_slot>(x, null)),
                     get_or<2>(t,if_else<2 == x_slot>(x, null)),
                     get_or<3>(t,if_else<3 == x_slot>(x, null)),
                     get_or<4>(t,if_else<4 == x_slot>(x, null)),
                     get_or<5>(t,if_else<5 == x_slot>(x, null)),
                     get_or<6>(t,if_else<6 == x_slot>(x, null)),
                     get_or<7>(t,if_else<7 == x_slot>(x, null)),
                     get_or<8>(t,if_else<8 == x_slot>(x, null)),
                     get_or<9>(t,if_else<9 == x_slot>(x, null)));
}


template<typename T, typename Tuple>
inline __host__ __device__
typename enable_if<
  tuple_size<Tuple>::value < 10,
  typename tuple_append_result<T,Tuple>::type
>::type
tuple_append(const Tuple &t, T &x)
{
  // the slot into which x will go
  static const int x_slot = thrust::tuple_size<Tuple>::value;

  typedef typename tuple_append_result<T,Tuple>::type result_type;

  // to produce an element of the tuple,
  // get the ith element of t, unless i is larger than
  // t's size
  // in that case, use x when i == x_slot,
  // otherwise use null_type
  
  const null_type null;

  return result_type(get_or<0>(t,if_else<0 == x_slot>(x, null)),
                     get_or<1>(t,if_else<1 == x_slot>(x, null)),
                     get_or<2>(t,if_else<2 == x_slot>(x, null)),
                     get_or<3>(t,if_else<3 == x_slot>(x, null)),
                     get_or<4>(t,if_else<4 == x_slot>(x, null)),
                     get_or<5>(t,if_else<5 == x_slot>(x, null)),
                     get_or<6>(t,if_else<6 == x_slot>(x, null)),
                     get_or<7>(t,if_else<7 == x_slot>(x, null)),
                     get_or<8>(t,if_else<8 == x_slot>(x, null)),
                     get_or<9>(t,if_else<9 == x_slot>(x, null)));
}


} // end tuple_detail
} // end detail
} // end thrust

