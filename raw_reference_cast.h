#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/reference.h>
#include <thrust/detail/type_traits/has_nested_type.h>
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/detail/type_traits.h>
#include <thrust/tuple.h>

namespace thrust
{
namespace detail
{

__THRUST_DEFINE_HAS_NESTED_TYPE(is_wrapped_reference, wrapped_reference_hint);

template<typename T, typename Enable = void>
  struct raw_reference
    : add_reference<T>
{};

template<typename T>
  struct raw_reference<
    T,
    typename thrust::detail::enable_if<
      is_wrapped_reference<
        typename remove_cv<T>::type
      >::value
    >::type
  >
    : add_reference<
        typename pointer_element<typename T::pointer>::type
      >
{};

// XXX raw_reference_tuple_helper might need to pass copies of values
//     rather than references to values
//     basically, we want the following behavior:
//     1. unwrap wrapped references to raw references
//     2. pass through raw references as raw references
//     3. pass through values as copies
//     4. pass through null_type as null_type
template<typename T>
  struct raw_reference_tuple_helper
    : raw_reference<T>
{};

template<>
  struct raw_reference_tuple_helper<null_type>
{
  typedef null_type type;
};


template <
  class T0, class T1, class T2,
  class T3, class T4, class T5,
  class T6, class T7, class T8,
  class T9
>
  struct raw_reference<
    thrust::tuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
  >
{
  typedef thrust::tuple<
    typename raw_reference_tuple_helper<T0>::type,
    typename raw_reference_tuple_helper<T1>::type,
    typename raw_reference_tuple_helper<T2>::type,
    typename raw_reference_tuple_helper<T3>::type,
    typename raw_reference_tuple_helper<T4>::type,
    typename raw_reference_tuple_helper<T5>::type,
    typename raw_reference_tuple_helper<T6>::type,
    typename raw_reference_tuple_helper<T7>::type,
    typename raw_reference_tuple_helper<T8>::type,
    typename raw_reference_tuple_helper<T9>::type
  > type;
};

} // end detail

template<typename T>
  __host__ __device__ typename detail::raw_reference<T>::type raw_reference_cast(T &ref)
{
  return *thrust::raw_pointer_cast(&ref);
}

template<typename T>
  __host__ __device__ typename detail::raw_reference<const T>::type raw_reference_cast(const T &ref)
{
  return *thrust::raw_pointer_cast(&ref);
}

template<typename T>
  __host__ __device__
  typename detail::raw_reference<thrust::tuple<T> >::type
    raw_reference_cast(const thrust::tuple<T> &t)
{
  std::cout << "raw_reference_cast(const tuple &): casting tuple" << std::endl;
  typedef typename detail::raw_reference<thrust::tuple<T> >::type result_type;
  return result_type(raw_reference_cast(thrust::get<0>(t)),
                     raw_reference_cast(thrust::get<1>(t)),
                     raw_reference_cast(thrust::get<2>(t)),
                     raw_reference_cast(thrust::get<3>(t)),
                     raw_reference_cast(thrust::get<4>(t)),
                     raw_reference_cast(thrust::get<5>(t)),
                     raw_reference_cast(thrust::get<6>(t)),
                     raw_reference_cast(thrust::get<7>(t)),
                     raw_reference_cast(thrust::get<8>(t)),
                     raw_reference_cast(thrust::get<9>(t)));
}

} // end thrust

