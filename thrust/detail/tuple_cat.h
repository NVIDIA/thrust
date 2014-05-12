#pragma once

#include <thrust/detail/config.h>
#include <thrust/tuple.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/tuple_helpers.h>

namespace thrust
{
namespace detail
{
namespace tuple_detail
{


// the type of concatenating two tuples
template<typename Tuple1, typename Tuple2, typename Enable = void>
struct tuple_cat_result2
  : tuple_cat_result2<
      typename tuple_append_result<     
        typename tuple_element<0,Tuple2>::type,
        Tuple1
      >::type,
      typename tuple_tail_result<
        Tuple2
      >::type
    >
{
};


// the type of concatenating a tuple with an empty tuple
template<typename Tuple1, typename Tuple2>
struct tuple_cat_result2<Tuple1, Tuple2, typename enable_if<tuple_size<Tuple2>::value == 0>::type>
  : identity_<Tuple1>
{};


template<typename Tuple1,
         typename Tuple2  = tuple<>,
         typename Tuple3  = tuple<>,
         typename Tuple4  = tuple<>,
         typename Tuple5  = tuple<>,
         typename Tuple6  = tuple<>,
         typename Tuple7  = tuple<>,
         typename Tuple8  = tuple<>,
         typename Tuple9  = tuple<>,
         typename Tuple10 = tuple<> >
class tuple_cat_result;


template<typename Tuple1, typename Tuple2, typename Tuple3, typename Tuple4,
         typename Tuple5, typename Tuple6, typename Tuple7, typename Tuple8,
         typename Tuple9, typename Tuple10>
struct tuple_cat_result
{
  private:
    typedef typename tuple_cat_result2<
      Tuple9, Tuple10
    >::type tuple9_10;
  
    typedef typename tuple_cat_result2<
      Tuple8, tuple9_10
    >::type tuple8_10;
  
    typedef typename tuple_cat_result2<
      Tuple7, tuple8_10
    >::type tuple7_10;
  
    typedef typename tuple_cat_result2<
      Tuple6, tuple7_10
    >::type tuple6_10;
  
    typedef typename tuple_cat_result2<
      Tuple5, tuple6_10
    >::type tuple5_10;
  
    typedef typename tuple_cat_result2<
      Tuple4, tuple5_10
    >::type tuple4_10;
  
    typedef typename tuple_cat_result2<
      Tuple3, tuple4_10
    >::type tuple3_10;
  
    typedef typename tuple_cat_result2<
      Tuple2, tuple3_10
    >::type tuple2_10;

  public:
    typedef typename tuple_cat_result2<
      Tuple1, tuple2_10
    >::type type;
};



} // end tuple_detail


template<typename Tuple1 = thrust::tuple<>,
         typename Tuple2 = thrust::tuple<>,
         typename Tuple3 = thrust::tuple<>,
         typename Tuple4 = thrust::tuple<>,
         typename Tuple5 = thrust::tuple<>,
         typename Tuple6 = thrust::tuple<>,
         typename Tuple7 = thrust::tuple<>,
         typename Tuple8 = thrust::tuple<>,
         typename Tuple9 = thrust::tuple<>,
         typename Tuple10 = thrust::tuple<> >
struct tuple_cat_enable_if
  : enable_if<
      (tuple_size<Tuple1>::value +
       tuple_size<Tuple2>::value +
       tuple_size<Tuple3>::value +
       tuple_size<Tuple4>::value +
       tuple_size<Tuple5>::value +
       tuple_size<Tuple6>::value +
       tuple_size<Tuple7>::value +
       tuple_size<Tuple8>::value +
       tuple_size<Tuple9>::value +
       tuple_size<Tuple10>::value)
      <= 10,
      typename tuple_detail::tuple_cat_result<Tuple1,Tuple2,Tuple3,Tuple4,Tuple5,Tuple6,Tuple7,Tuple8,Tuple9,Tuple10>::type
    >
{};


} // end detail


// terminal case of tuple_cat()
template<typename Tuple>
inline __host__ __device__
typename detail::tuple_cat_enable_if<Tuple>::type
  tuple_cat(const Tuple& t, const thrust::tuple<> &)
{
  return t;
}


template<typename Tuple1, typename Tuple2>
inline __host__ __device__
typename detail::tuple_cat_enable_if<Tuple1,Tuple2>::type
  tuple_cat(const Tuple1 &t1, const Tuple2 &t2)
{
  typedef typename thrust::tuple_element<0,Tuple2>::type head_type;

  // recurse by appending the head of t2 to t1
  // and concatenating the result with the tail of t2
  namespace ns = thrust::detail::tuple_detail;
  return thrust::tuple_cat(ns::tuple_append<head_type>(t1, thrust::get<0>(t2)), ns::tuple_tail(t2));
}


// XXX perhaps there's a smarter way to accumulate
template<typename Tuple1, typename Tuple2, typename Tuple3>
inline __host__ __device__
typename detail::tuple_cat_enable_if<Tuple1,Tuple2,Tuple3>::type
  tuple_cat(const Tuple1 &t1, const Tuple2 &t2, const Tuple3 &t3)
{
  return thrust::tuple_cat(t1, thrust::tuple_cat(t2,t3));
}


template<typename Tuple1, typename Tuple2, typename Tuple3, typename Tuple4>
inline __host__ __device__
typename detail::tuple_cat_enable_if<Tuple1,Tuple2,Tuple3,Tuple4>::type
  tuple_cat(const Tuple1 &t1, const Tuple2 &t2, const Tuple3 &t3, const Tuple4 &t4)
{
  return thrust::tuple_cat(t1, t2, thrust::tuple_cat(t3,t4));
}


template<typename Tuple1, typename Tuple2, typename Tuple3, typename Tuple4, typename Tuple5>
inline __host__ __device__
typename detail::tuple_cat_enable_if<Tuple1,Tuple2,Tuple3,Tuple4,Tuple5>::type
  tuple_cat(const Tuple1 &t1, const Tuple2 &t2, const Tuple3 &t3, const Tuple4 &t4, const Tuple5 &t5)
{
  return thrust::tuple_cat(t1, t2, t3, thrust::tuple_cat(t4,t5));
}


template<typename Tuple1, typename Tuple2, typename Tuple3, typename Tuple4, typename Tuple5, typename Tuple6>
inline __host__ __device__
typename detail::tuple_cat_enable_if<Tuple1,Tuple2,Tuple3,Tuple4,Tuple5,Tuple6>::type
  tuple_cat(const Tuple1 &t1, const Tuple2 &t2, const Tuple3 &t3, const Tuple4 &t4, const Tuple5 &t5, const Tuple6 &t6)
{
  return thrust::tuple_cat(t1, t2, t3, t4, thrust::tuple_cat(t5,t6));
}


template<typename Tuple1, typename Tuple2, typename Tuple3, typename Tuple4, typename Tuple5, typename Tuple6, typename Tuple7>
inline __host__ __device__
typename detail::tuple_cat_enable_if<Tuple1,Tuple2,Tuple3,Tuple4,Tuple5,Tuple6,Tuple7>::type
  tuple_cat(const Tuple1 &t1, const Tuple2 &t2, const Tuple3 &t3, const Tuple4 &t4, const Tuple5 &t5, const Tuple6 &t6, const Tuple7 &t7)
{
  return thrust::tuple_cat(t1, t2, t3, t4, t5, thrust::tuple_cat(t6,t7));
}


template<typename Tuple1, typename Tuple2, typename Tuple3, typename Tuple4, typename Tuple5, typename Tuple6, typename Tuple7, typename Tuple8>
inline __host__ __device__
typename detail::tuple_cat_enable_if<Tuple1,Tuple2,Tuple3,Tuple4,Tuple5,Tuple6,Tuple7,Tuple8>::type
  tuple_cat(const Tuple1 &t1, const Tuple2 &t2, const Tuple3 &t3, const Tuple4 &t4, const Tuple5 &t5, const Tuple6 &t6, const Tuple7 &t7, const Tuple8 &t8)
{
  return thrust::tuple_cat(t1, t2, t3, t4, t5, t6, thrust::tuple_cat(t7,t8));
}


template<typename Tuple1, typename Tuple2, typename Tuple3, typename Tuple4, typename Tuple5, typename Tuple6, typename Tuple7, typename Tuple8, typename Tuple9>
inline __host__ __device__
typename detail::tuple_cat_enable_if<Tuple1,Tuple2,Tuple3,Tuple4,Tuple5,Tuple6,Tuple7,Tuple8,Tuple9>::type
  tuple_cat(const Tuple2 &t1, const Tuple2 &t2, const Tuple3 &t3, const Tuple4 &t4, const Tuple5 &t5, const Tuple6 &t6, const Tuple7 &t7, const Tuple8 &t8, const Tuple9 &t9)
{
  return thrust::tuple_cat(t1, t2, t3, t4, t5, t6, t7, thrust::tuple_cat(t8,t9));
}


template<typename Tuple1, typename Tuple2, typename Tuple3, typename Tuple4, typename Tuple5, typename Tuple6, typename Tuple7, typename Tuple8, typename Tuple9, typename Tuple10>
inline __host__ __device__
typename detail::tuple_cat_enable_if<Tuple1,Tuple2,Tuple3,Tuple4,Tuple5,Tuple6,Tuple7,Tuple8,Tuple9,Tuple10>::type
  tuple_cat(const Tuple1 &t1, const Tuple2 &t2, const Tuple3 &t3, const Tuple4 &t4, const Tuple5 &t5, const Tuple6 &t6, const Tuple7 &t7, const Tuple8 &t8, const Tuple9 &t9, const Tuple10 &t10)
{
  return thrust::tuple_cat(t1, t2, t3, t4, t5, t6, t7, t8, thrust::tuple_cat(t9,t10));
}


} // end thrust

