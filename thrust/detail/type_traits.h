/*
 *  Copyright 2008-2011 NVIDIA Corporation
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


/*! \file type_traits.h
 *  \brief Temporarily define some type traits
 *         until nvcc can compile tr1::type_traits.
 */

#pragma once

#include <thrust/detail/config.h>

// XXX nvcc 2.2 closed beta can't compile type_traits
//// find type_traits
//
//#ifdef __GNUC__
//
//#if __GNUC__ == 4 && __GNUC_MINOR__ == 2
//#include <tr1/type_traits>
//#elif __GNUC__ == 4 && __GNUC_MINOR__ > 2
//#include <type_traits>
//#endif // GCC version
//
//#endif // GCC
//
//#ifdef _MSC_VER
//#include <type_traits>
//#endif // MSVC


namespace thrust
{

// forward declaration of device_reference
template<typename T> class device_reference;

namespace detail
{
 /// helper classes [4.3].
 template<typename _Tp, _Tp __v>
   struct integral_constant
   {
     static const _Tp                      value = __v;
     typedef _Tp                           value_type;
     typedef integral_constant<_Tp, __v>   type;
   };
 
 /// typedef for true_type
 typedef integral_constant<bool, true>     true_type;

 /// typedef for true_type
 typedef integral_constant<bool, false>    false_type;

//template<typename T> struct is_integral : public std::tr1::is_integral<T> {};
template<typename T> struct is_integral                           : public false_type {};
template<>           struct is_integral<bool>                     : public true_type {};
template<>           struct is_integral<char>                     : public true_type {};
template<>           struct is_integral<signed char>              : public true_type {};
template<>           struct is_integral<unsigned char>            : public true_type {};
template<>           struct is_integral<short>                    : public true_type {};
template<>           struct is_integral<unsigned short>           : public true_type {};
template<>           struct is_integral<int>                      : public true_type {};
template<>           struct is_integral<unsigned int>             : public true_type {};
template<>           struct is_integral<long>                     : public true_type {};
template<>           struct is_integral<unsigned long>            : public true_type {};
template<>           struct is_integral<long long>                : public true_type {};
template<>           struct is_integral<unsigned long long>       : public true_type {};
template<>           struct is_integral<const bool>               : public true_type {};
template<>           struct is_integral<const char>               : public true_type {};
template<>           struct is_integral<const unsigned char>      : public true_type {};
template<>           struct is_integral<const short>              : public true_type {};
template<>           struct is_integral<const unsigned short>     : public true_type {};
template<>           struct is_integral<const int>                : public true_type {};
template<>           struct is_integral<const unsigned int>       : public true_type {};
template<>           struct is_integral<const long>               : public true_type {};
template<>           struct is_integral<const unsigned long>      : public true_type {};
template<>           struct is_integral<const long long>          : public true_type {};
template<>           struct is_integral<const unsigned long long> : public true_type {};

template<typename T> struct is_floating_point              : public false_type {};
template<>           struct is_floating_point<float>       : public true_type {};
template<>           struct is_floating_point<double>      : public true_type {};
template<>           struct is_floating_point<long double> : public true_type {};

template<typename T> struct is_arithmetic               : public is_integral<T> {};
template<>           struct is_arithmetic<float>        : public true_type {};
template<>           struct is_arithmetic<double>       : public true_type {};
template<>           struct is_arithmetic<const float>  : public true_type {};
template<>           struct is_arithmetic<const double> : public true_type {};

template<typename T> struct is_pointer      : public false_type {};
template<typename T> struct is_pointer<T *> : public true_type  {};

template<typename T> struct is_device_ptr  : public false_type {};

template<typename T> struct is_void       : public false_type {};
template<>           struct is_void<void> : public true_type {};


namespace tt_detail
{


} // end tt_detail

template<typename T> struct is_pod
   : public integral_constant<
       bool,
       is_void<T>::value || is_pointer<T>::value || is_arithmetic<T>::value
#if THRUST_HOST_COMPILER   == THRUST_HOST_COMPILER_MSVC
// use intrinsic type traits
       || __is_pod(T)
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC
// only use the intrinsic for >= 4.3
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 3)
       || __is_pod(T)
#endif // GCC VERSION
#endif // THRUST_HOST_COMPILER
     >
 {};


template<typename T> struct has_trivial_constructor
  : public integral_constant<
      bool,
      is_pod<T>::value
#if THRUST_HOST_COMPILER   == THRUST_HOST_COMPILER_MSVC
      || __has_trivial_constructor(T)
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC
// only use the intrinsic for >= 4.3
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 3)
      || __has_trivial_constructor(T)
#endif // GCC VERSION
#endif // THRUST_HOST_COMPILER
      >
{};

template<typename T> struct has_trivial_copy_constructor
  : public integral_constant<
      bool,
      is_pod<T>::value
#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
      || __has_trivial_copy(T)
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC
// only use the intrinsic for >= 4.3
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 3)
      || __has_trivial_copy(T)
#endif // GCC VERSION
#endif // THRUST_HOST_COMPILER
    >
{};

template<typename T> struct has_trivial_destructor : public is_pod<T> {};

template<typename T> struct is_const          : public false_type {};
template<typename T> struct is_const<const T> : public true_type {};

template<typename T> struct is_volatile             : public false_type {};
template<typename T> struct is_volatile<volatile T> : public true_type {};

template<typename T>
  struct add_const
{
  typedef T const type;
}; // end add_const

template<typename T>
  struct remove_const
{
  typedef T type;
}; // end remove_const

template<typename T>
  struct remove_const<const T>
{
  typedef T type;
}; // end remove_const

template<typename T>
  struct add_volatile
{
  typedef volatile T type;
}; // end add_volatile

template<typename T>
  struct remove_volatile
{
  typedef T type;
}; // end remove_volatile

template<typename T>
  struct remove_volatile<volatile T>
{
  typedef T type;
}; // end remove_volatile

template<typename T>
  struct add_cv
{
  typedef const volatile T type;
}; // end add_cv

template<typename T>
  struct remove_cv
{
  typedef typename remove_const<typename remove_volatile<T>::type>::type type;
}; // end remove_cv


template<typename T> struct is_reference     : public false_type {};
template<typename T> struct is_reference<T&> : public true_type {};

template<typename T> struct is_device_reference                                : public false_type {};
template<typename T> struct is_device_reference< thrust::device_reference<T> > : public true_type {};


// NB: Careful with reference to void.
template<typename _Tp, bool = (is_void<_Tp>::value || is_reference<_Tp>::value)>
  struct __add_reference_helper
  { typedef _Tp&    type; };

template<typename _Tp>
  struct __add_reference_helper<_Tp, true>
  { typedef _Tp     type; };

template<typename _Tp>
  struct add_reference
    : public __add_reference_helper<_Tp>{};

template<typename T>
  struct remove_reference
{
  typedef T type;
}; // end remove_reference

template<typename T>
  struct remove_reference<T&>
{
  typedef T type;
}; // end remove_reference

template<typename T1, typename T2>
  struct is_same
    : public false_type
{
}; // end is_same

template<typename T>
  struct is_same<T,T>
    : public true_type
{
}; // end is_same

template<typename T1, typename T2>
  struct lazy_is_same
    : is_same<typename T1::type, typename T2::type>
{
}; // end lazy_is_same

template<typename T1, typename T2>
  struct is_different
    : public true_type
{
}; // end is_different

template<typename T>
  struct is_different<T,T>
    : public false_type
{
}; // end is_different

template<typename T1, typename T2>
  struct lazy_is_different
    : is_different<typename T1::type, typename T2::type>
{
}; // end lazy_is_different

namespace tt_detail
{

template<typename T>
  struct is_int_or_cref
{
  typedef typename remove_reference<T>::type type_sans_ref;
  static const bool value = (is_integral<T>::value
                             || (is_integral<type_sans_ref>::value
                                 && is_const<type_sans_ref>::value
                                 && !is_volatile<type_sans_ref>::value));
}; // end is_int_or_cref


__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN
__THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL_BEGIN


template<typename From, typename To>
  struct is_convertible_sfinae
{
  private:
    typedef char                          one_byte;
    typedef struct { char two_chars[2]; } two_bytes;

    static one_byte  test(To);
    static two_bytes test(...);
    static From      m_from;

  public:
    static const bool value = sizeof(test(m_from)) == sizeof(one_byte);
}; // end is_convertible_sfinae


__THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL_END
__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END


template<typename From, typename To>
  struct is_convertible_needs_simple_test
{
  static const bool from_is_void      = is_void<From>::value;
  static const bool to_is_void        = is_void<To>::value;
  static const bool from_is_float     = is_floating_point<typename remove_reference<From>::type>::value;
  static const bool to_is_int_or_cref = is_int_or_cref<To>::value;

  static const bool value = (from_is_void || to_is_void || (from_is_float && to_is_int_or_cref));
}; // end is_convertible_needs_simple_test


template<typename From, typename To,
         bool = is_convertible_needs_simple_test<From,To>::value>
  struct is_convertible
{
  static const bool value = (is_void<To>::value
                             || (is_int_or_cref<To>::value
                                 && !is_void<From>::value));
}; // end is_convertible


template<typename From, typename To>
  struct is_convertible<From, To, false>
{
  static const bool value = (is_convertible_sfinae<typename
                             add_reference<From>::type, To>::value);
}; // end is_convertible


} // end tt_detail

template<typename From, typename To>
  struct is_convertible
    : public integral_constant<bool, tt_detail::is_convertible<From, To>::value>
{
}; // end is_convertible


template<typename T1, typename T2>
  struct is_one_convertible_to_the_other
    : public integral_constant<
        bool,
        is_convertible<T1,T2>::value || is_convertible<T2,T1>::value
      >
{};


// mpl stuff

template <typename Condition1, typename Condition2, typename Condition3 = false_type>
  struct or_
    : public integral_constant<bool, Condition1::value || Condition2::value || Condition3::value>
{
}; // end or_

template <typename Condition1, typename Condition2>
  struct and_
    : public integral_constant<bool, Condition1::value && Condition2::value>
{
}; // end and_

template <typename Boolean>
  struct not_
    : public integral_constant<bool, !Boolean::value>
{
}; // end not_

template <bool, typename Then, typename Else>
  struct eval_if
{
}; // end eval_if

template<typename Then, typename Else>
  struct eval_if<true, Then, Else>
{
  typedef typename Then::type type;
}; // end eval_if

template<typename Then, typename Else>
  struct eval_if<false, Then, Else>
{
  typedef typename Else::type type;
}; // end eval_if

template<typename T>
//  struct identity
//  XXX WAR nvcc's confusion with thrust::identity
  struct identity_
{
  typedef T type;
}; // end identity

template<bool, typename T = void> struct enable_if {};
template<typename T>              struct enable_if<true, T> {typedef T type;};


template<typename T1, typename T2>
  struct enable_if_convertible
    : enable_if< is_convertible<T1,T2>::value>
{};


template<typename T>
  struct is_numeric
    : and_<
        is_convertible<int,T>,
        is_convertible<T,int>
      >
{
}; // end is_numeric


template<typename> struct is_reference_to_const             : false_type {};
template<typename T> struct is_reference_to_const<const T&> : true_type {};


// make_unsigned follows

namespace tt_detail
{

template<typename T> struct make_unsigned_simple;

template<> struct make_unsigned_simple<char>                   { typedef unsigned char          type; };
template<> struct make_unsigned_simple<signed char>            { typedef signed   char          type; };
template<> struct make_unsigned_simple<unsigned char>          { typedef unsigned char          type; };
template<> struct make_unsigned_simple<short>                  { typedef unsigned short         type; };
template<> struct make_unsigned_simple<unsigned short>         { typedef unsigned short         type; };
template<> struct make_unsigned_simple<int>                    { typedef unsigned int           type; };
template<> struct make_unsigned_simple<unsigned int>           { typedef unsigned int           type; };
template<> struct make_unsigned_simple<long int>               { typedef unsigned long int      type; };
template<> struct make_unsigned_simple<unsigned long int>      { typedef unsigned long int      type; };
template<> struct make_unsigned_simple<long long int>          { typedef unsigned long long int type; };
template<> struct make_unsigned_simple<unsigned long long int> { typedef unsigned long long int type; };

template<typename T>
  struct make_unsigned_base
{
  // remove cv
  typedef typename remove_cv<T>::type remove_cv_t;

  // get the simple unsigned type
  typedef typename make_unsigned_simple<remove_cv_t>::type unsigned_remove_cv_t;

  // add back const, volatile, both, or neither to the simple result
  typedef typename eval_if<
    is_const<T>::value && is_volatile<T>::value,
    // add cv back
    add_cv<unsigned_remove_cv_t>,
    // check const & volatile individually
    eval_if<
      is_const<T>::value,
      // add c back
      add_const<unsigned_remove_cv_t>,
      eval_if<
        is_volatile<T>::value,
        // add v back
        add_volatile<unsigned_remove_cv_t>,
        // original type was neither cv, return the simple unsigned result
        identity_<unsigned_remove_cv_t>
      >
    >
  >::type type;
};

} // end tt_detail

template<typename T>
  struct make_unsigned
    : tt_detail::make_unsigned_base<T>
{};

struct largest_available_float
{
#if defined(__CUDA_ARCH__)
#  if (__CUDA_ARCH__ < 130)
  typedef float type;
#  else
  typedef double type;
#  endif
#else
  typedef double type;
#endif
};

} // end detail

} // end thrust

#include <thrust/detail/type_traits/has_trivial_assign.h>

