/*
 *  Copyright 2008-2009 NVIDIA Corporation
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

#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/iterator_facade.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/detail/minimum_category.h>
#include <thrust/iterator/detail/minimum_space.h>
#include <thrust/tuple.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/device/dereference.h>

namespace thrust
{

// forward declare zip_iterator for zip_iterator_base
template<typename IteratorTuple> class zip_iterator;

// One important design goal of the zip_iterator is to isolate all
// functionality whose implementation relies on the current tuple
// implementation. This goal has been achieved as follows: Inside
// the namespace detail there is a namespace tuple_impl_specific.
// This namespace encapsulates all functionality that is specific
// to the current Boost tuple implementation. More precisely, the
// namespace tuple_impl_specific provides the following tuple
// algorithms and meta-algorithms for the current Boost tuple
// implementation:
//
// tuple_meta_transform
// tuple_meta_accumulate
// tuple_transform
// tuple_for_each
//
// If the tuple implementation changes, all that needs to be
// replaced is the implementation of these four (meta-)algorithms.

namespace detail
{

// Functors to be used with tuple algorithms
//
template<typename DiffType>
class advance_iterator
{
public:
  __host__ __device__
  advance_iterator(DiffType step) : m_step(step) {}
  
  template<typename Iterator>
  __host__ __device__
  void operator()(Iterator& it) const
  { it += m_step; }

private:
  DiffType m_step;
}; // end advance_iterator


struct increment_iterator
{
  template<typename Iterator>
  __host__ __device__
  void operator()(Iterator& it)
  { ++it; }
}; // end increment_iterator


struct decrement_iterator
{
  template<typename Iterator>
  __host__ __device__
  void operator()(Iterator& it)
  { --it; }
}; // end decrement_iterator


struct dereference_iterator
{
  template<typename Iterator>
  struct apply
  { 
    typedef typename
      iterator_traits<Iterator>::reference
    type;
  }; // end apply

  template<typename Iterator>
  __host__ __device__
    typename apply<Iterator>::type operator()(Iterator const& it)
  { return *it; }
}; // end dereference_iterator


struct device_dereference_iterator
{
  template<typename Iterator>
  struct apply
  { 
    typedef typename
      thrust::detail::iterator_device_reference<Iterator>::type
    type;
  }; // end apply

  template<typename Iterator>
  __device__
    typename apply<Iterator>::type operator()(Iterator const& it)
  { return ::thrust::detail::device::dereference(it); }
}; // end device_dereference_iterator


template<typename IndexType>
struct device_dereference_iterator_with_index
{
  template<typename Iterator>
  struct apply
  { 
    typedef typename
      thrust::detail::iterator_device_reference<Iterator>::type
    type;
  }; // end apply

  template<typename Iterator>
  __device__
    typename apply<Iterator>::type operator()(Iterator const& it)
  { return ::thrust::detail::device::dereference(it, n); }

  IndexType n;
}; // end device_dereference_iterator


// The namespace tuple_impl_specific provides two meta-
// algorithms and two algorithms for tuples.
namespace tuple_impl_specific
{

// define apply1 for tuple_meta_transform_impl
template<typename UnaryMetaFunctionClass, class Arg>
  struct apply1
    : UnaryMetaFunctionClass::template apply<Arg>
{
}; // end apply1


// implement support for extremely simple lambda expressions

// if X is not a placeholder expression, lambda returns X unchanged as type
template<typename X>
  struct lambda
{
  typedef X type;
}; // end lambda

// if X is a placeholder expression, lambda returns a type which can evaluate X as type
template< template <typename> class X >
  struct lambda< X<_1> >
{
  // type has a member, apply, which applies X to an argument
  struct type
  {
    template <typename Arg>
      struct apply
    {
      typedef typename X<Arg>::type type;
    }; // end apply
  }; // end type
}; // end lambda


// Meta-transform algorithm for tuples

// forward declaraction of tuple_meta_transform for tuple_meta_transform_impl
template<typename Tuple, class UnaryMetaFun>
  struct tuple_meta_transform;

template<typename Tuple, class UnaryMetaFun>
  struct tuple_meta_transform_impl
{
    typedef thrust::detail::cons<
        typename apply1<
            // XXX do we need to implement mpl::lambda or not?
            //typename mpl::lambda<UnaryMetaFun>::type
            typename lambda<UnaryMetaFun>::type
            //UnaryMetaFun
          , typename Tuple::head_type
        >::type
      , typename tuple_meta_transform<
            typename Tuple::tail_type
          , UnaryMetaFun 
        >::type
    > type;
};

template<typename Tuple, class UnaryMetaFun>
  struct tuple_meta_transform
    : thrust::detail::eval_if<
          thrust::detail::is_same<Tuple, thrust::null_type>::value
        , thrust::detail::identity_<thrust::null_type>
        , tuple_meta_transform_impl<Tuple, UnaryMetaFun>
      >
{
};



// define apply2 for tuple_meta_accumulate_impl
template<typename UnaryMetaFunctionClass, class Arg1, class Arg2>
  struct apply2
    : UnaryMetaFunctionClass::template apply<Arg1,Arg2>
{
}; // end apply2


// Meta-accumulate algorithm for tuples. Note: The template 
// parameter StartType corresponds to the initial value in 
// ordinary accumulation.
//
template<class Tuple, class BinaryMetaFun, class StartType>
  struct tuple_meta_accumulate;

template<
    typename Tuple
  , class BinaryMetaFun
  , typename StartType
>
  struct tuple_meta_accumulate_impl
{
   typedef typename apply2<
       // XXX do we need to implement mpl::lambda or not?
       //typename mpl::lambda<BinaryMetaFun>::type
       BinaryMetaFun
     , typename Tuple::head_type
     , typename tuple_meta_accumulate<
           typename Tuple::tail_type
         , BinaryMetaFun
         , StartType 
       >::type
   >::type type;
};


template<
    typename Tuple
  , class BinaryMetaFun
  , typename StartType
>
struct tuple_meta_accumulate
  : thrust::detail::eval_if<
        thrust::detail::is_same<Tuple, thrust::null_type>::value
      , thrust::detail::identity_<StartType>
      , tuple_meta_accumulate_impl<
            Tuple
          , BinaryMetaFun
          , StartType
        >
    > // end eval_if
{
}; // end tuple_meta_accumulate


// transform algorithm for tuples. The template parameter Fun
// must be a unary functor which is also a unary metafunction
// class that computes its return type based on its argument
// type. For example:
//
// struct to_ptr
// {
//     template <class Arg>
//     struct apply
//     {
//          typedef Arg* type;
//     }
//
//     template <class Arg>
//     Arg* operator()(Arg x);
// };
template<typename Fun>
__host__ __device__
thrust::null_type tuple_transform
    (thrust::null_type const&, Fun)
{ return thrust::null_type(); }



template<typename Tuple, typename Fun>
__host__ __device__
typename tuple_meta_transform<
    Tuple
  , Fun
>::type

tuple_transform(
  const Tuple& t, 
  Fun f
)
{ 
    typedef typename tuple_meta_transform<
        typename Tuple::tail_type
      , Fun
    >::type transformed_tail_type;

  return thrust::detail::cons<
      typename apply1<
          Fun, typename Tuple::head_type
       >::type
     , transformed_tail_type
  >( 
      f(thrust::get<0>(t)), tuple_transform(t.get_tail(), f)
  );
} // end tuple_transform()



// for_each algorithm for tuples.
//
template<typename Fun>
__host__ __device__
Fun tuple_for_each(thrust::null_type, Fun f)
{
  return f;
} // end tuple_for_each()


template<typename Tuple, typename Fun>
__host__ __device__
Fun tuple_for_each(Tuple& t, Fun f)
{ 
    f( t.get_head() );
    return tuple_for_each(t.get_tail(), f);
} // end tuple_for_each()



// Equality of tuples. NOTE: "==" for tuples currently (7/2003)
// has problems under some compilers, so I just do my own.
// No point in bringing in a bunch of #ifdefs here. This is
// going to go away with the next tuple implementation anyway.
//
__host__ __device__
inline bool tuple_equal(thrust::null_type, thrust::null_type)
{ return true; }


template<typename Tuple1, typename Tuple2>
__host__ __device__
bool tuple_equal(Tuple1 const& t1, Tuple2 const& t2)
{ 
  return t1.get_head() == t2.get_head() && 
  tuple_equal(t1.get_tail(), t2.get_tail());
} // end tuple_equal()

} // end end tuple_impl_specific


// define the lambda placeholders for the metafunctions below
struct _1 {};
struct _2 {};


// specialize iterator_reference on the lambda placeholder
template<typename T> struct
  iterator_reference
    : thrust::iterator_reference<T>
{
}; // end iterator_reference

template<>
  struct iterator_reference<_1>
{
  template <class T>
    struct apply : thrust::iterator_reference<T> {};
}; // end iterator_reference


// specialize iterator_device_reference on the lambda placeholder
template<>
  struct iterator_device_reference<_1>
{
  template <class T>
    struct apply : thrust::detail::iterator_device_reference<T> {};
}; // end iterator_device_reference

namespace zip_iterator_base_ns
{

// specialize iterator_value on the lambda placeholder
template<typename T>
  struct iterator_value
    : thrust::iterator_value<T>
{
}; // end iterator_value

template<>
  struct iterator_value<_1>
{
  template <class T>
    struct apply : thrust::iterator_value<T> {};
}; // end iterator_value

} // end zip_iterator_base_ns


// Metafunction to obtain the type of the tuple whose element types
// are the reference types of an iterator tuple.
//
template<typename IteratorTuple>
  struct tuple_of_references
    : tuple_impl_specific::tuple_meta_transform<
          IteratorTuple, 
          iterator_reference<_1>
        >
{
}; // end tuple_of_references


// Metafunction to obtain the type of the tuple whose element types
// are the device reference types of an iterator tuple.
template<typename IteratorTuple>
  struct tuple_of_device_references
    : tuple_impl_specific::tuple_meta_transform<
          IteratorTuple,
          iterator_device_reference<_1>
        >
{
}; // end tuple_of_device_references


// Metafunction to obtain the type of the tuple whose element types
// are the value_types of an iterator tupel.
//
template<typename IteratorTuple>
  struct tuple_of_value_types
    : tuple_impl_specific::tuple_meta_transform<
          IteratorTuple,
          zip_iterator_base_ns::iterator_value<_1>
        >
{
}; // end tuple_of_value_types



// Metafunction to obtain the minimal traversal tag in a tuple
// of iterators.
//
template<typename IteratorTuple>
struct minimum_traversal_category_in_iterator_tuple
{
  typedef typename tuple_impl_specific::tuple_meta_transform<
      IteratorTuple
    , thrust::iterator_traversal<_1>
  >::type tuple_of_traversal_tags;
      
  typedef typename tuple_impl_specific::tuple_meta_accumulate<
      tuple_of_traversal_tags
    , minimum_category<>
    , thrust::random_access_traversal_tag
  >::type type;
};



// Metafunction to obtain the minimal space tag in a tuple
// of iterators.
template<typename IteratorTuple>
struct minimum_space_in_iterator_tuple
{
  typedef typename tuple_impl_specific::tuple_meta_transform<
    IteratorTuple,
    thrust::iterator_space<_1>
  >::type tuple_of_space_tags;

  typedef typename tuple_impl_specific::tuple_meta_accumulate<
    tuple_of_space_tags,
    minimum_space<>,
    thrust::any_space_tag
  >::type type;
};

  
//// We need to call tuple_meta_accumulate with mpl::and_ as the
//// accumulating functor. To this end, we need to wrap it into
//// a struct that has exactly two arguments (that is, template
//// parameters) and not five, like mpl::and_ does.
////
//template<typename Arg1, typename Arg2>
//struct and_with_two_args
//  : mpl::and_<Arg1, Arg2>
//{
//};
    


///////////////////////////////////////////////////////////////////
//
// Class zip_iterator_base
//
// Builds and exposes the iterator facade type from which the zip 
// iterator will be derived.
//
template<typename IteratorTuple>
  struct zip_iterator_base
{
 private:
    // reference type is the type of the tuple obtained from the
    // iterators' reference types.
    typedef typename tuple_of_references<IteratorTuple>::type reference;

    // Boost's Value type is the same as reference type.
    //typedef reference value_type;
    typedef typename tuple_of_value_types<IteratorTuple>::type value_type;

    // Boost's Pointer type is just value_type *
    //typedef value_type * pointer;
    typedef reference * pointer;

    // Difference type is the first iterator's difference type
    typedef typename thrust::iterator_traits<
      typename thrust::tuple_element<0, IteratorTuple>::type
    >::difference_type difference_type;

    // Iterator space is the minimum space tag in the
    // iterator tuple
    typedef typename
    minimum_space_in_iterator_tuple<IteratorTuple>::type space;

    // Traversal category is the minimum traversal category in the
    // iterator tuple
    typedef typename
    minimum_traversal_category_in_iterator_tuple<IteratorTuple>::type traversal_category;
  
 public:
  
    // The iterator facade type from which the zip iterator will
    // be derived.
    typedef experimental::iterator_facade<
        zip_iterator<IteratorTuple>,
        pointer,
        value_type,  
        space,
        traversal_category,
        reference,
        difference_type
    > type;
}; // end zip_iterator_base

} // end detail

} // end thrust


