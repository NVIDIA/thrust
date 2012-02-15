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


/*! \file transform_iterator.h
 *  \brief An iterator which adapts another iterator by applying a function to the result of its dereference 
 */

/*
 * (C) Copyright David Abrahams 2002.
 * (C) Copyright Jeremy Siek    2002.
 * (C) Copyright Thomas Witt    2002.
 * 
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
 */

#pragma once

#include <thrust/detail/config.h>

// #include the details first
#include <thrust/iterator/detail/transform_iterator.inl>
#include <thrust/iterator/iterator_facade.h>
#include <thrust/detail/type_traits.h>

namespace thrust
{

/*! \addtogroup iterators
 *  \{
 */

/*! \addtogroup fancyiterator Fancy Iterators
 *  \ingroup iterators
 *  \{
 */

/*! \p transform_iterator is an iterator which represents a pointer into a range
 *  of values after transformation by a function. This iterator is useful for 
 *  creating a range filled with the result of applying an operation to another range
 *  without either explicitly storing it in memory, or explicitly executing the transformation.
 *  Using \p transform_iterator facilitates kernel fusion by deferring the execution
 *  of a transformation until the value is needed while saving both memory capacity
 *  and bandwidth.
 *
 *  The following code snippet demonstrates how to create a \p transform_iterator
 *  which represents the result of \c sqrtf applied to the contents of a \p device_vector.
 *
 *  \code
 *  #include <thrust/iterator/transform_iterator.h>
 *  #include <thrust/device_vector.h>
 *  
 *  // note: functor inherits from unary_function
 *  struct square_root : public thrust::unary_function<float,float>
 *  {
 *    __host__ __device__
 *    float operator()(float x) const
 *    {
 *      return sqrtf(x);
 *    }
 *  };
 *  
 *  int main(void)
 *  {
 *      thrust::device_vector<float> v(4);
 *      v[0] = 1.0f;
 *      v[1] = 4.0f;
 *      v[2] = 9.0f;
 *      v[3] = 16.0f;
 *                                                                                             
 *      typedef thrust::device_vector<float>::iterator FloatIterator;
 *                                                                                             
 *      thrust::transform_iterator<square_root, FloatIterator> iter(v.begin(), square_root());
 *                                                                                             
 *      *iter;   // returns 1.0f
 *      iter[0]; // returns 1.0f;
 *      iter[1]; // returns 2.0f;
 *      iter[2]; // returns 3.0f;
 *      iter[3]; // returns 4.0f;
 *                                                                                             
 *      // iter[4] is an out-of-bounds error
 *  }
 *  \endcode
 *
 *  This next example demonstrates how to use a \p transform_iterator with the
 *  \p thrust::reduce function to compute the sum of squares of a sequence.
 *  We will create temporary \p transform_iterators with the
 *  \p make_transform_iterator function in order to avoid explicitly specifying their type:
 *
 *  \code
 *  #include <thrust/iterator/transform_iterator.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/reduce.h>
 *  #include <iostream>
 *  
 *  // note: functor inherits from unary_function
 *  struct square : public thrust::unary_function<float,float>
 *  {
 *    __host__ __device__
 *    float operator()(float x) const
 *    {
 *      return x * x;
 *    }
 *  };
 *  
 *  int main(void)
 *  {
 *    // initialize a device array
 *    thrust::device_vector<float> v(4);
 *    v[0] = 1.0f;
 *    v[1] = 2.0f;
 *    v[2] = 3.0f;
 *    v[3] = 4.0f;
 *  
 *    float sum_of_squares =
 *     thrust::reduce(thrust::make_transform_iterator(v.begin(), square()),
 *                    thrust::make_transform_iterator(v.end(),   square()));
 *  
 *    std::cout << "sum of squares: " << sum_of_squares << std::endl;
 *    return 0;
 *  }
 *  \endcode
 *
 *  Note that in the previous two examples the transform functor (namely \c square_root 
 *  and \c square) inherits from \c thrust::unary_function.  Inheriting from 
 *  \c thrust::unary_function ensures that a functor is a valid \c AdaptableUnaryFunction
 *  and provides all the necessary \c typedef declarations.  The \p transform_iterator
 *  can also be applied to a \c UnaryFunction that does not inherit from 
 *  \c thrust::unary_function using an optional template argument.  The following example
 *  illustrates how to use the third template argument to specify the \c result_type of
 *  the function.   
 *
 *  \code
 *  #include <thrust/iterator/transform_iterator.h>
 *  #include <thrust/device_vector.h>
 *  
 *  // note: functor *does not* inherit from unary_function
 *  struct square_root
 *  {
 *    __host__ __device__
 *    float operator()(float x) const
 *    {
 *      return sqrtf(x);
 *    }
 *  };
 *  
 *  int main(void)
 *  {
 *      thrust::device_vector<float> v(4);
 *      v[0] = 1.0f;
 *      v[1] = 4.0f;
 *      v[2] = 9.0f;
 *      v[3] = 16.0f;
 *                                                                                             
 *      typedef thrust::device_vector<float>::iterator FloatIterator;
 *      
 *      // note: float result_type is specified explicitly
 *      thrust::transform_iterator<square_root, FloatIterator, float> iter(v.begin(), square_root());
 *                                                                                             
 *      *iter;   // returns 1.0f
 *      iter[0]; // returns 1.0f;
 *      iter[1]; // returns 2.0f;
 *      iter[2]; // returns 3.0f;
 *      iter[3]; // returns 4.0f;
 *                                                                                             
 *      // iter[4] is an out-of-bounds error
 *  }
 *  \endcode
 *
 *  \see make_transform_iterator
 */
template <class AdaptableUnaryFunction, class Iterator, class Reference = use_default, class Value = use_default>
  class transform_iterator
    : public detail::transform_iterator_base<AdaptableUnaryFunction, Iterator, Reference, Value>::type
{
  /*! \cond
   */
  public:
    typedef typename
    detail::transform_iterator_base<AdaptableUnaryFunction, Iterator, Reference, Value>::type
    super_t;

    friend class experimental::iterator_core_access;
  /*! \endcond
   */

  public:
    /*! Null constructor does nothing.
     */
    __host__ __device__
    transform_iterator() {}
  
    /*! This constructor takes as arguments an \c Iterator and an \c AdaptableUnaryFunction
     *  and copies them to a new \p transform_iterator.
     *
     *  \param x An \c Iterator pointing to the input to this \p transform_iterator's \c AdaptableUnaryFunction.
     *  \param f An \c AdaptableUnaryFunction used to transform the objects pointed to by \p x.
     */
    __host__ __device__
    transform_iterator(Iterator const& x, AdaptableUnaryFunction f)
      : super_t(x), m_f(f) {
    }
  
    /*! This explicit constructor copies the value of a given \c Iterator and creates
     *  this \p transform_iterator's \c AdaptableUnaryFunction using its null constructor.
     *
     *  \param x An \c Iterator to copy.
     */
    __host__ __device__
    explicit transform_iterator(Iterator const& x)
      : super_t(x) { }

    /*! This copy constructor creates a new \p transform_iterator from another
     *  \p transform_iterator.
     *
     *  \param other The \p transform_iterator to copy.
     */
    template<typename OtherAdaptableUnaryFunction,
             typename OtherIterator,
             typename OtherReference,
             typename OtherValue>
    __host__ __device__
    transform_iterator(const transform_iterator<OtherAdaptableUnaryFunction, OtherIterator, OtherReference, OtherValue> &other,
                       typename thrust::detail::enable_if_convertible<OtherIterator, Iterator>::type* = 0,
                       typename thrust::detail::enable_if_convertible<OtherAdaptableUnaryFunction, AdaptableUnaryFunction>::type* = 0)
      : super_t(other.base()), m_f(other.functor()) {}

    /*! This method returns a copy of this \p transform_iterator's \c AdaptableUnaryFunction.
     *  \return A copy of this \p transform_iterator's \c AdaptableUnaryFunction.
     */
    __host__ __device__
    AdaptableUnaryFunction functor() const
      { return m_f; }

    /*! \cond
     */
  private:
    typename super_t::reference dereference() const
    { 
      return m_f(*this->base());
    }

    // tag this as mutable per Dave Abrahams in this thread:
    // http://lists.boost.org/Archives/boost/2004/05/65332.php
    mutable AdaptableUnaryFunction m_f;

    /*! \endcond
     */
}; // end transform_iterator


/*! \p make_transform_iterator creates a \p transform_iterator
 *  from an \c Iterator and \c AdaptableUnaryFunction.
 *
 *  \param it The \c Iterator pointing to the input range of the
 *            newly created \p transform_iterator.
 *  \param fun The \c AdaptableUnaryFunction used to transform the range pointed
 *             bo by \p it in the newly created \p transform_iterator.
 *  \return A new \p transform_iterator which transforms the range at
 *          \p it by \p fun.
 *  \see transform_iterator
 */
template <class AdaptableUnaryFunction, class Iterator>
inline __host__ __device__
transform_iterator<AdaptableUnaryFunction, Iterator>
make_transform_iterator(Iterator it, AdaptableUnaryFunction fun)
{
  return transform_iterator<AdaptableUnaryFunction, Iterator>(it, fun);
} // end make_transform_iterator

/*! \} // end fancyiterators
 */

/*! \} // end iterators
 */

} // end thrust

