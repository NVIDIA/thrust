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
#include <thrust/detail/temporary_array.h>
#include <thrust/system/detail/internal/scalar/merge.h>
#include <thrust/system/detail/internal/scalar/insertion_sort.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace internal
{
namespace scalar
{
namespace detail
{

template <typename RandomAccessIterator,
          typename StrictWeakOrdering>
void inplace_merge(RandomAccessIterator first,
                   RandomAccessIterator middle,
                   RandomAccessIterator last,
                   StrictWeakOrdering comp)
{
  // XXX the type of exec should be:
  //     typedef decltype(select_system(first, middle, last)) DerivedPolicy;
  typedef typename thrust::iterator_system<RandomAccessIterator>::type DerivedPolicy;
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;

  // XXX assumes DerivedPolicy is default constructible
  // XXX find a way to get a stateful execution policy into this function
  //     or simply pass scratch space
  DerivedPolicy exec;
  thrust::detail::temporary_array<value_type, DerivedPolicy> a(exec, first, middle);
  thrust::detail::temporary_array<value_type, DerivedPolicy> b(exec, middle, last);

  thrust::system::detail::internal::scalar::merge(a.begin(), a.end(), b.begin(), b.end(), first, comp);
}

template <typename RandomAccessIterator1,
          typename RandomAccessIterator2,
          typename StrictWeakOrdering>
void inplace_merge_by_key(RandomAccessIterator1 first1,
                          RandomAccessIterator1 middle1,
                          RandomAccessIterator1 last1,
                          RandomAccessIterator2 first2,
                          StrictWeakOrdering comp)
{
  // XXX the type of exec should be:
  //     typedef decltype(select_system(first1, middle1, last1, first2)) DerivedPolicy;
  typedef typename thrust::iterator_system<RandomAccessIterator1>::type DerivedPolicy;
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type1;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type value_type2;

  RandomAccessIterator2 middle2 = first2 + (middle1 - first1);
  RandomAccessIterator2 last2   = first2 + (last1   - first1);

  // XXX assumes DerivedPolicy is default constructible
  // XXX find a way to get a stateful exec into this function
  //     or simply pass scratch space
  DerivedPolicy exec;
  thrust::detail::temporary_array<value_type1, DerivedPolicy> lhs1(exec, first1, middle1);
  thrust::detail::temporary_array<value_type1, DerivedPolicy> rhs1(exec, middle1, last1);
  thrust::detail::temporary_array<value_type2, DerivedPolicy> lhs2(exec, first2, middle2);
  thrust::detail::temporary_array<value_type2, DerivedPolicy> rhs2(exec, middle2, last2);

  thrust::system::detail::internal::scalar::merge_by_key
    (lhs1.begin(), lhs1.end(), rhs1.begin(), rhs1.end(),
     lhs2.begin(), rhs2.begin(),
     first1, first2, comp);
}

} // end namespace detail

//////////////
// Key Sort //
//////////////

template <typename RandomAccessIterator,
          typename StrictWeakOrdering>
void stable_merge_sort(RandomAccessIterator first,
                       RandomAccessIterator last,
                       StrictWeakOrdering comp)
{
  if (last - first < 32)
  {
    thrust::system::detail::internal::scalar::insertion_sort(first, last, comp);
  }
  else
  {
    RandomAccessIterator middle = first + (last - first) / 2;

    thrust::system::detail::internal::scalar::stable_merge_sort(first, middle, comp);
    thrust::system::detail::internal::scalar::stable_merge_sort(middle,  last, comp);
    detail::inplace_merge(first, middle, last, comp);
  }
}


////////////////////
// Key-Value Sort //
////////////////////

template <typename RandomAccessIterator1,
          typename RandomAccessIterator2,
          typename StrictWeakOrdering>
void stable_merge_sort_by_key(RandomAccessIterator1 first1,
                              RandomAccessIterator1 last1,
                              RandomAccessIterator2 first2,
                              StrictWeakOrdering comp)
{
  if (last1 - first1 <= 32)
  {
    thrust::system::detail::internal::scalar::insertion_sort_by_key(first1, last1, first2, comp);
  }
  else
  {
    RandomAccessIterator1 middle1 = first1 + (last1 - first1) / 2;
    RandomAccessIterator2 middle2 = first2 + (last1 - first1) / 2;

    thrust::system::detail::internal::scalar::stable_merge_sort_by_key(first1, middle1, first2,  comp);
    thrust::system::detail::internal::scalar::stable_merge_sort_by_key(middle1,  last1, middle2, comp);
    detail::inplace_merge_by_key(first1, middle1, last1, first2, comp);
  }
}

} // end namespace scalar
} // end namespace internal
} // end namespace detail
} // end namespace system
} // end namespace thrust

