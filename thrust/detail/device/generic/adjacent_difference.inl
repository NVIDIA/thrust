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


#include <thrust/iterator/iterator_traits.h>

#include <thrust/device_ptr.h>
#include <thrust/detail/raw_buffer.h>
#include <thrust/copy.h>
#include <thrust/transform.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace generic
{
namespace detail
{

// TODO remove this when adjacent_difference supports in-place operation
template <typename Iterator1, typename Iterator2>
bool is_same_iterator(Iterator1, Iterator2)
{
  return false;
}
template <typename Iterator1>
bool is_same_iterator(Iterator1 iter1, Iterator1 iter2)
{
  return iter1 == iter2;
}

} // end namespace detail


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
    else if(detail::is_same_iterator(first, result))
    {
        // an in-place operation is requested, copy the input and call the entry point
        // XXX a special-purpose kernel would be faster here since
        // only block boundaries need to be copied
        thrust::detail::raw_buffer<InputType, Space> input_copy(first, last);
        thrust::detail::device::generic::adjacent_difference(input_copy.begin(), input_copy.end(), result, binary_op);
    }
    else
    {
        // XXX a special-purpose kernel would be faster here
        *result = *first;
        thrust::transform(first + 1, last, first, result + 1, binary_op); 
    } // end else

    return result + (last - first);
}

} // end namespace generic
} // end namespace device
} // end namespace detail
} // end namespace thrust

