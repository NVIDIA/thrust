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

// this iterator's implementation is based on zip_iterator, so we grab all of its includes
#include <thrust/iterator/detail/zip_iterator_base.h>

//
// nvcc cannot compile the line from below:
//
//     return thrust::detail::tuple_host_device_transform<detail::tuple_dereference_iterator<Iterator>::template apply>(i, detail::tuple_dereference_iterator<Iterator>(this->_it));
//
// it incorrectly reports:
//
//     error: dependent-name 'thrust::detail::tuple_dereference_iterator::apply' is parsed as a non-type, but instantiation yields a type
//     note: say 'typename thrust::detail::tuple_dereference_iterator::apply' if a type is meant
//
// To see that this a compiler bug, compare the analogous line from zip_iterator_base.h:
// since detail::dereference_iterator is not a template, and thus thrust::detail::tuple_dereference_iterator::apply 
// is not a dependent-name, nvcc is able to compile it correctly
//
// to workaround this, we reimplement tuple_host_device_transform
// in a way that avoids passing this dependent-name-template-template apply in (TupleMetaFunction)
// and instead just specify XfrmTuple directly
//
#ifdef __CUDACC__
#  define WAR_NVCC_CANNOT_HANDLE_DEPENDENT_TEMPLATE_TEMPLATE_ARGUMENT
#endif


#ifdef WAR_NVCC_CANNOT_HANDLE_DEPENDENT_TEMPLATE_TEMPLATE_ARGUMENT
#  include <thrust/iterator/detail/multi_permutation_iterator_tuple_transform.h>
#endif

namespace thrust
{

template<typename,typename> class multi_permutation_iterator;

namespace detail
{

template<typename Substitute, typename T>
struct tuple_substitution
{
    typedef Substitute type;
};

template<typename Substitute>
    struct tuple_substitution<Substitute, thrust::null_type>
{
    typedef thrust::null_type type;
};
    
template<typename Substitute, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
struct tuple_substitution <Substitute, thrust::tuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9> >
{
    typedef thrust::tuple<
        typename tuple_substitution<Substitute,T0>::type,
        typename tuple_substitution<Substitute,T1>::type,
        typename tuple_substitution<Substitute,T2>::type,
        typename tuple_substitution<Substitute,T3>::type,
        typename tuple_substitution<Substitute,T4>::type,
        typename tuple_substitution<Substitute,T5>::type,
        typename tuple_substitution<Substitute,T6>::type,
        typename tuple_substitution<Substitute,T7>::type,
        typename tuple_substitution<Substitute,T8>::type,
        typename tuple_substitution<Substitute,T9>::type
    > type;
}; // end of tuple_substitution

template<typename Substitute, typename T>
struct tuple_reference_substitution
{
    typedef Substitute type;
};

template<typename Substitute>
    struct tuple_reference_substitution<Substitute, thrust::null_type>
{
    typedef thrust::null_type type;
};

template<typename Substitute, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
struct tuple_reference_substitution <Substitute, thrust::tuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9> >
{
    typedef thrust::detail::tuple_of_iterator_references<
        typename tuple_reference_substitution<Substitute,T0>::type,
        typename tuple_reference_substitution<Substitute,T1>::type,
        typename tuple_reference_substitution<Substitute,T2>::type,
        typename tuple_reference_substitution<Substitute,T3>::type,
        typename tuple_reference_substitution<Substitute,T4>::type,
        typename tuple_reference_substitution<Substitute,T5>::type,
        typename tuple_reference_substitution<Substitute,T6>::type,
        typename tuple_reference_substitution<Substitute,T7>::type,
        typename tuple_reference_substitution<Substitute,T8>::type,
        typename tuple_reference_substitution<Substitute,T9>::type
    > type;
}; // end of tuple_reference_substitution

// Metafunction to obtain a tuple whose element types 
// are all the reference type of ElementIterator,
// where the tuple is the same size as IndexTupleIterator
template<typename ElementIterator, typename IndexTupleIterator>
  struct indexed_tuple_of_references
    : tuple_reference_substitution<typename iterator_traits<ElementIterator>::reference, typename thrust::iterator_traits<IndexTupleIterator>::value_type>
{
}; // end indexed_tuple_of_references


// Metafunction to obtain a tuple whose element types 
// are all the value type of ElementIterator,
// where the tuple is the same size as IndexTupleIterator
template<typename ElementIterator, typename IndexTupleIterator>
  struct indexed_tuple_of_value_types
    : tuple_substitution<typename iterator_traits<ElementIterator>::value_type, typename thrust::iterator_traits<IndexTupleIterator>::value_type>
{
}; // end indexed_tuple_of_value_types

template<typename ElementIterator,
         typename IndexTupleIterator>
struct multi_permutation_iterator_base
{
    typedef typename thrust::iterator_system<   ElementIterator>::type System1;
    typedef typename thrust::iterator_system<IndexTupleIterator>::type System2;

    //private:
    // reference type is the type of the tuple obtained from the
    // iterators' reference types.
    typedef typename indexed_tuple_of_references<ElementIterator,IndexTupleIterator>::type reference;
    
    // Boost's Value type is the same as reference type.
    //typedef reference value_type;
    typedef typename indexed_tuple_of_value_types<ElementIterator,IndexTupleIterator>::type value_type;
    
    // Boost's Pointer type is just value_type *
    //typedef value_type * pointer;
    typedef reference * pointer;
    
    // Difference type is the IndexTuple iterator's difference type
    typedef typename thrust::iterator_traits<
        IndexTupleIterator
    >::difference_type difference_type;
    
    // Iterator system is the minimum system tag in the
    // iterator tuple
    typedef typename
    detail::minimum_system<System1,System2>::type system;

public:
    
    typedef thrust::experimental::iterator_adaptor<
    multi_permutation_iterator<ElementIterator,IndexTupleIterator>,
    IndexTupleIterator,
    pointer,
    value_type,  
    system,
    thrust::use_default,
    reference
    > type;
}; // end multi_permutation_iterator_base    

template<typename Iterator>
struct tuple_dereference_iterator
{
  Iterator _it;
  __host__ __device__ tuple_dereference_iterator(Iterator it) : _it(it) {}

  template<typename Offset>
  struct apply
  { 
    typedef typename
      tuple_reference_substitution<typename thrust::iterator_traits<Iterator>::reference, Offset>::type
    type;
  }; // end apply

  template<typename Offset>
  __host__ __device__
  typename apply<Offset>::type operator()(Offset const& i)
  { return *(this->_it + i); }

  template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
  __host__ __device__
  typename apply<thrust::tuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9> >::type operator()(thrust::tuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9> const& i)
  {
#ifndef WAR_NVCC_CANNOT_HANDLE_DEPENDENT_TEMPLATE_TEMPLATE_ARGUMENT      
    return thrust::detail::                                               tuple_host_device_transform<detail::tuple_dereference_iterator<Iterator>::template apply       >(i, detail::tuple_dereference_iterator<Iterator>(this->_it));
#else
    return thrust::detail::multi_permutation_iterator_tuple_transform_ns::tuple_host_device_transform<typename apply<thrust::tuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9> >::type>(i, detail::tuple_dereference_iterator<Iterator>(this->_it));      
#endif
  }
}; // end tuple_dereference_iterator

} // end detail

} // end thrust

