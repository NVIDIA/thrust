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

#include <thrust/iterator/multi_permutation_iterator.h>

namespace thrust
{

template <typename ElementIterator, typename IndexTupleIterator>
  typename multi_permutation_iterator<ElementIterator, IndexTupleIterator>::super_t::reference
    multi_permutation_iterator<ElementIterator, IndexTupleIterator>
      ::dereference(void) const
{
  typename thrust::iterator_traits<IndexTupleIterator>::value_type index_tuple = *(this->base());   // this converts from reference to value, and avoids issues with device_reference not being a tuple when dispatcing the host_device_transform below

#ifndef WAR_NVCC_CANNOT_HANDLE_DEPENDENT_TEMPLATE_TEMPLATE_ARGUMENT      
  return thrust::detail::tuple_host_device_transform<detail::tuple_dereference_iterator<ElementIterator>::template apply>(index_tuple, detail::tuple_dereference_iterator<ElementIterator>(m_element_iterator));
#else
  return thrust::detail::multi_permutation_iterator_tuple_transform_ns::tuple_host_device_transform< typename multi_permutation_iterator<ElementIterator, IndexTupleIterator>::super_t::reference >(index_tuple, detail::tuple_dereference_iterator<ElementIterator>(m_element_iterator));
#endif
} // end multi_permutation_iterator::dereference()

} // end thrust

