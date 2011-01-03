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

#include <thrust/iterator/reverse_iterator.h>
#include <thrust/detail/device/dereference.h>
#include <thrust/iterator/iterator_traits.h>

namespace thrust
{

namespace detail
{

template<typename Iterator>
__host__ __device__
  Iterator prior(Iterator x)
{
  return --x;
} // end prior()

} // end detail

template<typename BidirectionalIterator>
  reverse_iterator<BidirectionalIterator>
    ::reverse_iterator(BidirectionalIterator x)
      :super_t(x)
{
} // end reverse_iterator::reverse_iterator()

template<typename BidirectionalIterator>
  template<typename OtherBidirectionalIterator>
    reverse_iterator<BidirectionalIterator>
      ::reverse_iterator(reverse_iterator<OtherBidirectionalIterator> const &r
// XXX msvc screws this up
#ifndef _MSC_VER
                     , typename thrust::detail::enable_if<
                         thrust::detail::is_convertible<
                           OtherBidirectionalIterator,
                           BidirectionalIterator
                         >::value
                       >::type *
#endif // _MSC_VER
                     )
        :super_t(r.base())
{
} // end reverse_iterator::reverse_iterator()

template<typename BidirectionalIterator>
  typename reverse_iterator<BidirectionalIterator>::super_t::reference
    reverse_iterator<BidirectionalIterator>
      ::dereference(void) const
{
  return *thrust::detail::prior(this->base());
} // end reverse_iterator::increment()

template<typename BidirectionalIterator>
  void reverse_iterator<BidirectionalIterator>
    ::increment(void)
{
  --this->base_reference();
} // end reverse_iterator::increment()

template<typename BidirectionalIterator>
  void reverse_iterator<BidirectionalIterator>
    ::decrement(void)
{
  ++this->base_reference();
} // end reverse_iterator::decrement()

template<typename BidirectionalIterator>
  void reverse_iterator<BidirectionalIterator>
    ::advance(typename super_t::difference_type n)
{
  this->base_reference() += -n;
} // end reverse_iterator::advance()

template<typename BidirectionalIterator>
  template<typename OtherBidirectionalIterator>
    typename reverse_iterator<BidirectionalIterator>::super_t::difference_type
      reverse_iterator<BidirectionalIterator>
        ::distance_to(reverse_iterator<OtherBidirectionalIterator> const &y) const
{
  return this->base_reference() - y.base();
} // end reverse_iterator::distance_to()

template<typename BidirectionalIterator>
__host__ __device__
reverse_iterator<BidirectionalIterator> make_reverse_iterator(BidirectionalIterator x)
{
  return reverse_iterator<BidirectionalIterator>(x);
} // end make_reverse_iterator()


namespace detail
{

namespace device
{

template<typename DeviceBidirectionalIterator>
  struct dereference_result< thrust::reverse_iterator<DeviceBidirectionalIterator> >
    : dereference_result<DeviceBidirectionalIterator>
{
}; // end dereference_result

template<typename BidirectionalIterator>
  inline __host__ __device__
    typename dereference_result< thrust::reverse_iterator<BidirectionalIterator> >::type
      dereference(const thrust::reverse_iterator<BidirectionalIterator> &iter)
{
  return dereference(thrust::detail::prior(iter.base()));
} // end dereference()

template<typename BidirectionalIterator, typename IndexType>
  inline __host__ __device__
    typename dereference_result< thrust::reverse_iterator<BidirectionalIterator> >::type
      dereference(const thrust::reverse_iterator<BidirectionalIterator> &iter, IndexType n)
{
  thrust::reverse_iterator<BidirectionalIterator> temp = iter;
  temp += n;
  return dereference(temp);
} // end dereference()

} // end device

} // end detail

} // end thrust

