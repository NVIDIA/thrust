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
#include <thrust/system/tbb/detail/tag.h>
#include <thrust/system/detail/internal/scalar/merge.h>
#include <thrust/system/detail/internal/scalar/binary_search.h>
#include <tbb/parallel_for.h>

namespace thrust
{
namespace system
{
namespace tbb
{
namespace detail
{
namespace merge_detail
{

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
struct range
{
  InputIterator1 first1, last1;
  InputIterator2 first2, last2;
  OutputIterator result;
  StrictWeakOrdering comp;
  size_t grain_size;

  range(InputIterator1 first1, InputIterator1 last1,
        InputIterator2 first2, InputIterator2 last2,
        OutputIterator result,
        StrictWeakOrdering comp,
        size_t grain_size = 1024)
    : first1(first1), last1(last1),
      first2(first2), last2(last2),
      result(result), comp(comp), grain_size(grain_size)
  {}
  
  range(range& r, ::tbb::split)
    : first1(r.first1), last1(r.last1),
      first2(r.first2), last2(r.last2),
      result(r.result), comp(r.comp), grain_size(r.grain_size)
  {
    // we can assume n1 and n2 are not both 0
    size_t n1 = thrust::distance(first1, last1);
    size_t n2 = thrust::distance(first2, last2);

    InputIterator1 mid1 = first1;
    InputIterator2 mid2 = first2;

    if (n1 > n2)
    {
      mid1 += n1 / 2;
      mid2 = thrust::system::detail::internal::scalar::lower_bound(first2, last2, raw_reference_cast(*mid1), comp);
    }
    else
    {
      mid2 += n2 / 2;
      mid1 = thrust::system::detail::internal::scalar::upper_bound(first1, last1, raw_reference_cast(*mid2), comp);
    }
    
    // set first range to [first1, mid1), [first2, mid2), result
    r.last1 = mid1;
    r.last2 = mid2;

    // set second range to [mid1, last1), [mid2, last2), result + (mid1 - first1) + (mid2 - first2)
    first1 = mid1;
    first2 = mid2;
    result += thrust::distance(r.first1, mid1) + thrust::distance(r.first2, mid2);
  }

  bool empty(void) const
  {
    return (first1 == last1) && (first2 == last2);
  }

  bool is_divisible(void) const
  {
    return static_cast<size_t>(thrust::distance(first1, last1) + thrust::distance(first2, last2)) > grain_size;
  }
};

struct body
{
  template <typename Range>
  void operator()(Range& r) const
  {
    thrust::system::detail::internal::scalar::merge
      (r.first1, r.last1,
       r.first2, r.last2,
       r.result,
       r.comp);
  }
};

} // end namespace merge_detail

namespace merge_by_key_detail
{

template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2,
         typename StrictWeakOrdering>
struct range
{
  InputIterator1 first1, last1;
  InputIterator2 first2, last2;
  InputIterator3 first3;
  InputIterator4 first4;
  OutputIterator1 output1;
  OutputIterator2 output2;
  StrictWeakOrdering comp;
  size_t grain_size;

  range(InputIterator1 first1, InputIterator1 last1,
        InputIterator2 first2, InputIterator2 last2,
        InputIterator3 first3,
        InputIterator4 first4,
        OutputIterator1 output1,
        OutputIterator2 output2,
        StrictWeakOrdering comp,
        size_t grain_size = 1024)
    : first1(first1), last1(last1),
      first2(first2), last2(last2),
      first3(first3),
      first4(first4),
      output1(output1), output2(output2),
      comp(comp), grain_size(grain_size)
  {}
  
  range(range& r, ::tbb::split)
    : first1(r.first1), last1(r.last1),
      first2(r.first2), last2(r.last2),
      first3(r.first3),
      first4(r.first4),
      output1(r.output1), output2(r.output2),
      comp(r.comp), grain_size(r.grain_size)
  {
    // we can assume n1 and n2 are not both 0
    size_t n1 = thrust::distance(first1, last1);
    size_t n2 = thrust::distance(first2, last2);

    InputIterator1 mid1 = first1;
    InputIterator2 mid2 = first2;

    if (n1 > n2)
    {
      mid1 += n1 / 2;
      mid2 = thrust::system::detail::internal::scalar::lower_bound(first2, last2, raw_reference_cast(*mid1), comp);
    }
    else
    {
      mid2 += n2 / 2;
      mid1 = thrust::system::detail::internal::scalar::upper_bound(first1, last1, raw_reference_cast(*mid2), comp);
    }
    
    // set first range to [first1, mid1), [first2, mid2), output1, output2
    r.last1 = mid1;
    r.last2 = mid2;

    // set second range to [mid1, last1), [mid2, last2), output1 + (mid1 - first1) + (mid2 - first2), output2 + (mid1 - first1) + (mid2 - first2) 
    first1 = mid1;
    first2 = mid2;
    first3 += thrust::distance(r.first1, mid1);
    first4 += thrust::distance(r.first2, mid2);
    output1 += thrust::distance(r.first1, mid1) + thrust::distance(r.first2, mid2);
    output2 += thrust::distance(r.first1, mid1) + thrust::distance(r.first2, mid2);
  }

  bool empty(void) const
  {
    return (first1 == last1) && (first2 == last2);
  }

  bool is_divisible(void) const
  {
    return static_cast<size_t>(thrust::distance(first1, last1) + thrust::distance(first2, last2)) > grain_size;
  }
};

struct body
{
  template <typename Range>
  void operator()(Range& r) const
  {
    thrust::system::detail::internal::scalar::merge_by_key
      (r.first1, r.last1,
       r.first2, r.last2,
       r.first3,
       r.first4,
       r.output1,
       r.output2,
       r.comp);
  }
};

} // end namespace merge_by_key_detail


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
OutputIterator merge(tag,
                     InputIterator1 first1,
                     InputIterator1 last1,
                     InputIterator2 first2,
                     InputIterator2 last2,
                     OutputIterator result,
                     StrictWeakOrdering comp)
{
  typedef typename merge_detail::range<InputIterator1,InputIterator2,OutputIterator,StrictWeakOrdering> Range;
  typedef          merge_detail::body                                                                   Body;
  Range range(first1, last1, first2, last2, result, comp);
  Body  body;

  ::tbb::parallel_for(range, body);

  thrust::advance(result, thrust::distance(first1, last1) + thrust::distance(first2, last2));

  return result;
} // end merge()

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakOrdering>
thrust::pair<OutputIterator1,OutputIterator2>
    merge_by_key(tag,
                 InputIterator1 first1,
                 InputIterator1 last1,
                 InputIterator2 first2,
                 InputIterator2 last2,
                 InputIterator3 first3,
                 InputIterator4 first4,
                 OutputIterator1 output1,
                 OutputIterator2 output2,
                 StrictWeakOrdering comp)
{
  typedef typename merge_by_key_detail::range<InputIterator1,InputIterator2,InputIterator3,InputIterator4,OutputIterator1,OutputIterator2,StrictWeakOrdering> Range;
  typedef          merge_by_key_detail::body                                                                                                                  Body;

  Range range(first1, last1, first2, last2, first3, first4, output1, output2, comp);
  Body  body;

  ::tbb::parallel_for(range, body);

  thrust::advance(output1, thrust::distance(first1, last1) + thrust::distance(first2, last2));
  thrust::advance(output2, thrust::distance(first1, last1) + thrust::distance(first2, last2));

  return thrust::make_pair(output1,output2);
}

} // end namespace detail
} // end namespace tbb
} // end namespace system
} // end namespace thrust

