// Copyright (c) 2017-2018 NVIDIA Corporation
// Copyright (c) 2014-2018 Bryce Adelstein Lelbach
// Copyright (c) 2001-2015 Housemarque Oy (housemarque.com)
// Copyright (c) 2007-2015 Hartmut Kaiser
// Copyright (c)      2002 Peter Dimov and Multi Media Ltd
//                         (`THRUST_CURRENT_FUNCTION`)
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#pragma once

///////////////////////////////////////////////////////////////////////////////

/// \def THRUST_PP_STRINGIZE(expr)
/// \brief Stringizes the expression \a expr.
///
/// \par <b>Example</b>:
///
/// \code
/// #include <iostream>
///
/// int main()
/// {
///   std::cout << THRUST_PP_STRINGIZE(foo) << std::endl;
/// }
/// \endcode
///
/// The above code expands to:
///
/// \code
/// #include <iostream>
///
/// int main()
/// {
///   std::cout << "foo" << std::endl;
/// }
/// \endcode
///
#define THRUST_PP_STRINGIZE_(expr) #expr
#define THRUST_PP_STRINGIZE(expr)  THRUST_PP_STRINGIZE_(expr)

///////////////////////////////////////////////////////////////////////////////

/// \def THRUST_PP_CAT2(a, b)
/// \brief Concatenates the tokens \a a and \b b.
///
/// \par <b>Example</b>:
///
/// \code
/// #include <iostream>
///
/// int main()
/// {
///   std::cout << THRUST_PP_CAT2(1, THRUST_PP_CAT2(2, 3)) << std::endl;
/// }
/// \endcode
///
/// The above code expands to:
///
/// \code
/// #include <iostream>
///
/// int main()
/// {
///   std::cout << 123 << std::endl;
/// }
/// \endcode
///
#define THRUST_PP_CAT2(a, b) THRUST_PP_CAT2_IMPL(a, b)

#if    defined(_MSC_VER)                                                      \
    && (defined(__EDG__) || defined(__EDG_VERSION__))                         \
    && (defined(__INTELLISENSE__) || __EDG_VERSION__ >= 308)
    #define THRUST_PP_CAT2_IMPL(a, b) THRUST_PP_CAT2_IMPL2(~, a ## b)
    #define THRUST_PP_CAT2_IMPL2(p, res) res
#else
    #define THRUST_PP_CAT2_IMPL(a, b) a ## b
#endif

///////////////////////////////////////////////////////////////////////////////

/// \def THRUST_PP_EXPAND(x)
/// \brief Performs macro expansion on \a x.
///
/// \par <b>Example</b>:
///
/// \code
/// #include <iostream>
///
/// #define FOO_BAR() "foo_bar"
/// #define BUZZ()     THRUST_PP_EXPAND(THRUST_PP_CAT2(FOO_, BAR)())
///
/// int main()
/// {
///   std::cout << BUZZ() << std::endl;
/// }
/// \endcode
///
/// The above code expands to:
///
/// \code
/// #include <iostream>
///
/// int main()
/// {
///   std::cout << "foo_bar" << std::endl;
/// }
/// \endcode
///
#define THRUST_PP_EXPAND(x) x

///////////////////////////////////////////////////////////////////////////////

/// \def THRUST_PP_ARITY(...)
/// \brief Returns the number of arguments that it was called with. Must be
///        called with less than 64 arguments.
///
/// \par <b>Example</b>:
///
/// \code
/// #include <iostream>
///
/// int main()
/// {
///   std::cout << THRUST_PP_ARITY()        << std::endl
///             << THRUST_PP_ARITY(x)       << std::endl
///             << THRUST_PP_ARITY(x, y)    << std::endl
///             << THRUST_PP_ARITY(x, y, z) << std::endl;
/// }
/// \endcode
///
/// The above code expands to:
///
/// \code
/// #include <iostream>
///
/// int main()
/// {
///   std::cout << 0 << std::endl
///             << 1 << std::endl
///             << 2 << std::endl
///             << 3 << std::endl;
/// }
/// \endcode
///
#define THRUST_PP_ARITY(...)                                                  \
  THRUST_PP_EXPAND(THRUST_PP_ARITY_IMPL(__VA_ARGS__,                          \
  63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48,                            \
  47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32,                            \
  31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,                            \
  15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))                           \
  /**/

#define THRUST_PP_ARITY_IMPL(                                                 \
   _1, _2, _3, _4, _5, _6, _7, _8, _9,_10,_11,_12,_13,_14,_15,_16,            \
  _17,_18,_19,_20,_21,_22,_23,_24,_25,_26,_27,_28,_29,_30,_31,_32,            \
  _33,_34,_35,_36,_37,_38,_39,_40,_41,_42,_43,_44,_45,_46,_47,_48,            \
  _49,_50,_51,_52,_53,_54,_55,_56,_57,_58,_59,_60,_61,_62,_63,  N,...) N      \
  /**/

/// \def THRUST_PP_DISPATCH(basename, ...)
/// \brief Expands to <tt>basenameN(...)</tt>, where <tt>N</tt> is the number
///        of variadic arguments that \a THRUST_PP_DISPATCH was called with.
///        This macro can be used to implement "macro overloading".
///
/// \par <b>Example</b>:
///
/// \code
/// #include <iostream>
///
/// #define PLUS(...) THRUST_PP_DISPATCH(PLUS, __VA_ARGS__)
/// #define PLUS1(x)       x
/// #define PLUS2(x, y)    x + y
/// #define PLUS3(x, y, z) x + y + z
///
/// int main()
/// {
///   std::cout << PLUS(1)       << std::endl
///             << PLUS(1, 2)    << std::endl
///             << PLUS(1, 2, 3) << std::endl;
/// }
/// \endcode
///
/// The above code expands to:
///
/// \code
/// #include <iostream>
///
/// #define PLUS(...) THRUST_PP_DISPATCH(PLUS, __VA_ARGS__)
/// #define PLUS1(x)       x
/// #define PLUS2(x, y)    x + y
/// #define PLUS3(x, y, z) x + y + z
///
/// int main()
/// {
///   std::cout << 1         << std::endl
///             << 1 + 2     << std::endl
///             << 1 + 2 + 3 << std::endl;
/// }
/// \endcode
///
#define THRUST_PP_DISPATCH(basename, ...)                                     \
  THRUST_PP_EXPAND(THRUST_PP_CAT2(basename,                                   \
    THRUST_PP_ARITY(__VA_ARGS__))(__VA_ARGS__))                               \
  /**/

///////////////////////////////////////////////////////////////////////////////

/// \def THRUST_CURRENT_FUNCTION
/// \brief The name of the current function as a string.
///
#if    defined(__GNUC__)                                                      \
    || (defined(__MWERKS__) && (__MWERKS__ >= 0x3000))                        \
    || (defined(__ICC) && (__ICC >= 600)) || defined(__ghs__)
  #define THRUST_CURRENT_FUNCTION __PRETTY_FUNCTION__
#elif defined(__DMC__) && (__DMC__ >= 0x810)
  #define THRUST_CURRENT_FUNCTION __PRETTY_FUNCTION__
#elif defined(__FUNCSIG__)
  #define THRUST_CURRENT_FUNCTION __FUNCSIG__
#elif    (defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 600))             \
      || (defined(__IBMCTHRUST_PP__) && (__IBMCTHRUST_PP__ >= 500))
  #define THRUST_CURRENT_FUNCTION __FUNCTION__
#elif defined(__BORLANDC__) && (__BORLANDC__ >= 0x550)
  #define THRUST_CURRENT_FUNCTION __FUNC__
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)
  #define THRUST_CURRENT_FUNCTION __func__
#elif defined(__cplusplus) && (__cplusplus >= 201103)
  #define THRUST_CURRENT_FUNCTION __func__
#else
  #define THRUST_CURRENT_FUNCTION "(unknown)"
#endif

///////////////////////////////////////////////////////////////////////////////

