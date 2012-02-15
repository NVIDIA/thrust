/*
 *  Copyright 2008-2012 NVIDIA Corporation
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

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/function_traits.h>
#include <thrust/detail/type_traits/iterator/is_output_iterator.h>

namespace thrust
{

namespace detail
{

// this trait reports what type should be used as a temporary in certain algorithms
// which aggregate intermediate results from a function before writing to an output iterator

// the pseudocode for deducing the type of the temporary used below:
// 
// if Function is an AdaptableFunction
//   result = Function::result_type
// else if OutputIterator2 is a "pure" output iterator
//   result = InputIterator2::value_type
// else
//   result = OutputIterator2::value_type
//
// XXX upon c++0x, TemporaryType needs to be:
// result_of<BinaryFunction>::type
template<typename InputIterator, typename OutputIterator, typename Function>
  struct intermediate_type_from_function_and_iterators
    : eval_if<
        has_result_type<Function>::value,
        result_type<Function>,
        eval_if<
          is_output_iterator<OutputIterator>::value,
          thrust::iterator_value<InputIterator>,
          thrust::iterator_value<OutputIterator>
        >
      >
{
}; // end intermediate_type_from_function_and_iterators

} // end detail

} // end thrust

