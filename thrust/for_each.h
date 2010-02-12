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


/*! \file for_each.h
 *  \brief Defines the interface to for_each.
 */

#pragma once

#include <thrust/detail/config.h>

namespace thrust
{

/*! \addtogroup modifying
 *  \ingroup transformations
 *  \{
 */

/*! \p for_each applies the function object \p f to each element
 *  in the range <tt>[first, last)</tt>; \p f's return value, if any,
 *  is ignored. Unlike the C++ Standard Template Library function
 *  <tt>std::for_each</tt>, this version offers no guarantee on
 *  order of execution. For this reason, this version of \p for_each
 *  has no return value.
 *
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *  \param f The function object to apply to the range <tt>[first, last)</tt>.
 *
 *  \tparam InputIterator is a model of <a href="http://www.sgi.com/tech/stl/InputIterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is convertible to \p UnaryFunction's \c argument_type.
 *  \tparam UnaryFunction is a model of <a href="http://www.sgi.com/tech/stl/UnaryFunction">Unary Function</a>,
 *          and \p UnaryFunction does not apply any non-constant operation through its argument.
 *
 *  \see http://www.sgi.com/tech/stl/for_each.html
 */
template<typename InputIterator,
         typename UnaryFunction>
void for_each(InputIterator first,
              InputIterator last,
              UnaryFunction f);

/*! \} // end modifying
 */

} // end namespace thrust

#include <thrust/detail/for_each.inl>

