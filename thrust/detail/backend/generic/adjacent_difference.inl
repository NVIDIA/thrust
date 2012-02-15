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


#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/uninitialized_array.h>
#include <thrust/transform.h>

namespace thrust
{
namespace detail
{
namespace backend
{
namespace generic
{

template <class InputIterator, class OutputIterator, class BinaryFunction>
OutputIterator adjacent_difference(InputIterator first, InputIterator last,
                                   OutputIterator result,
                                   BinaryFunction binary_op)
{
    typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;
    typedef typename thrust::iterator_space<InputIterator>::type Space;

    if(first == last)
    {
        // empty range, nothing to do
        return result; 
    }
    else 
    {
        // an in-place operation is requested, copy the input and call the entry point
        // XXX a special-purpose kernel would be faster here since
        // only block boundaries need to be copied
        thrust::detail::uninitialized_array<InputType, Space> input_copy(first, last);
        
        *result = *first;
        thrust::transform(input_copy.begin() + 1, input_copy.end(), input_copy.begin(), result + 1, binary_op); 
    }

    return result + (last - first);
}

} // end namespace generic
} // end namespace backend
} // end namespace detail
} // end namespace thrust

