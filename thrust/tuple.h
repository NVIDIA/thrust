/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

/*! \file tuple.h
 *  \brief A type encapsulating a heterogeneous collection of elements.
 */

/*
 * Copyright (C) 1999, 2000 Jaakko Järvi (jaakko.jarvi@cs.utu.fi)
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
 */

#pragma once

#include <thrust/detail/config.h>

#include <tuple>

#include <cuda/std/tuple>
#include <cuda/std/utility>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup utility
 *  \{
 */

/*! \addtogroup tuple
 *  \{
 */

/*! This metafunction returns the type of a
 *  \p tuple's <tt>N</tt>th element.
 *
 *  \tparam N This parameter selects the element of interest.
 *  \tparam T A \c tuple type of interest.
 *
 *  \see pair
 *  \see tuple
 */
template <size_t N, class T>
using tuple_element = ::cuda::std::tuple_element<N, T>;

/*! This metafunction returns the number of elements
 *  of a \p tuple type of interest.
 *
 *  \tparam T A \c tuple type of interest.
 *
 *  \see pair
 *  \see tuple
 */
template <class T>
using tuple_size = ::cuda::std::tuple_size<T>;

/*! \brief \p tuple is a class template that can be instantiated with up to ten
 *  arguments. Each template argument specifies the type of element in the \p
 *  tuple. Consequently, tuples are heterogeneous, fixed-size collections of
 *  values. An instantiation of \p tuple with two arguments is similar to an
 *  instantiation of \p pair with the same two arguments. Individual elements
 *  of a \p tuple may be accessed with the \p get function.
 *
 *  \tparam TN The type of the <tt>N</tt> \c tuple element. Thrust's \p tuple
 *          type currently supports up to ten elements.
 *
 *  The following code snippet demonstrates how to create a new \p tuple object
 *  and inspect and modify the value of its elements.
 *
 *  \code
 *  #include <thrust/tuple.h>
 *  #include <iostream>
 *
 *  int main() {
 *    // Create a tuple containing an `int`, a `float`, and a string.
 *    thrust::tuple<int, float, const char*> t(13, 0.1f, "thrust");
 *
 *    // Individual members are accessed with the free function `get`.
 *    std::cout << "The first element's value is " << thrust::get<0>(t) << std::endl;
 *
 *    // ... or the member function `get`.
 *    std::cout << "The second element's value is " << t.get<1>() << std::endl;
 *
 *    // We can also modify elements with the same function.
 *    thrust::get<0>(t) += 10;
 *  }
 *  \endcode
 *
 *  \see pair
 *  \see get
 *  \see make_tuple
 *  \see tuple_element
 *  \see tuple_size
 *  \see tie
 */
template <class... T>
using tuple = ::cuda::std::tuple<T...>;

using ::cuda::std::get;
using ::cuda::std::make_tuple;
using ::cuda::std::tie;

/*! \endcond
 */

/*! \} // tuple
 */

/*! \} // utility
 */

THRUST_NAMESPACE_END
