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


/*! \file for_each.inl
 *  \brief Inline file for for_each.h.
 */

#include <thrust/detail/backend/for_each.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{

namespace detail
{


template<typename OutputIterator,
         typename Size,
         typename UnaryFunction>
  OutputIterator for_each_n(OutputIterator first,
                            Size n,
                            UnaryFunction f)
{
  return thrust::detail::backend::for_each_n(first, n, f);
} // end for_each_n()

template<typename InputIterator,
         typename UnaryFunction>
  InputIterator for_each(InputIterator first,
                         InputIterator last,
                         UnaryFunction f)
{
  return thrust::detail::backend::for_each(first, last, f);
} // end for_each()


} // end detail


/////////////////
// Entry Point //
/////////////////
template<typename InputIterator,
         typename UnaryFunction>
void for_each(InputIterator first,
              InputIterator last,
              UnaryFunction f)
{
  thrust::detail::for_each(first, last, f);
} // end for_each()


} // end namespace thrust

