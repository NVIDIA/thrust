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


/*! \file functional.h
 *  \brief Function objects and tools for manipulating them
 */

#pragma once

#include <thrust/detail/config.h>
#include <functional>
#include <thrust/detail/functional/placeholder.h>

namespace thrust
{

/*! \addtogroup function_objects Function Objects
 */

template<typename Operation> struct unary_traits;

template<typename Operation> struct binary_traits;

/*! \addtogroup function_object_adaptors Function Object Adaptors
 *  \ingroup function_objects
 *  \{
 */

/*! \p unary_function is an empty base class: it contains no member functions
 *  or member variables, but only type information. The only reason it exists
 *  is to make it more convenient to define types that are models of the
 *  concept Adaptable Unary Function. Specifically, any model of Adaptable
 *  Unary Function must define nested \c typedefs. Those \c typedefs are
 *  provided by the base class \p unary_function.
 *
 *  The following code snippet demonstrates how to construct an 
 *  Adaptable Unary Function using \p unary_function.
 *
 *  \code
 *  struct sine : public thrust::unary_function<float,float>
 *  {
 *    __host__ __device__
 *    float operator()(float x) { return sinf(x); }
 *  };
 *  \endcode
 *
 *  \note unary_function is currently redundant with the C++ STL type
 *  \c std::unary_function. We reserve it here for potential additional
 *  functionality at a later date.
 *
 *  \see http://www.sgi.com/tech/stl/unary_function.html
 *  \see binary_function
 */
template<typename Argument,
         typename Result>
  struct unary_function
    : public std::unary_function<Argument, Result>
{
}; // end unary_function

/*! \p binary_function is an empty base class: it contains no member functions
 *  or member variables, but only type information. The only reason it exists
 *  is to make it more convenient to define types that are models of the
 *  concept Adaptable Binary Function. Specifically, any model of Adaptable
 *  Binary Function must define nested \c typedefs. Those \c typedefs are
 *  provided by the base class \p binary_function.
 *
 *  The following code snippet demonstrates how to construct an 
 *  Adaptable Binary Function using \p binary_function.
 *
 *  \code
 *  struct exponentiate : public thrust::binary_function<float,float,float>
 *  {
 *    __host__ __device__
 *    float operator()(float x, float y) { return powf(x,y); }
 *  };
 *  \endcode
 *
 *  \note binary_function is currently redundant with the C++ STL type
 *  \c std::binary_function. We reserve it here for potential additional
 *  functionality at a later date.
 *
 *  \see http://www.sgi.com/tech/stl/binary_function.html
 *  \see unary_function
 */
template<typename Argument1,
         typename Argument2,
         typename Result>
  struct binary_function
    : public std::binary_function<Argument1, Argument2, Result>
{
}; // end binary_function

/*! \}
 */


/*! \addtogroup predefined_function_objects Predefined Function Objects
 *  \ingroup function_objects
 */

/*! \addtogroup arithmetic_operations Arithmetic Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

/*! \p plus is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>plus<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x+y</tt>.
 *
 *  \tparam T is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x+y</tt> must be defined and must have a return type that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>plus</tt> to sum two
 *  device_vectors of \c floats.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *  thrust::device_vector<float> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 75);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                     thrust::plus<float>());
 *  // V3 is now {76, 77, 78, ..., 1075}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/plus.html
 *  \see binary_function
 */
template<typename T>
  struct plus : public binary_function<T,T,T>
{
  /*! Function call operator. The return value is <tt>lhs + rhs</tt>.
   */
  __host__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs + rhs;}
}; // end plus

/*! \p minus is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>minus<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x-y</tt>.
 *
 *  \tparam T is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x-y</tt> must be defined and must have a return type that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>minus</tt> to subtract
 *  a device_vector of \c floats from another.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *  thrust::device_vector<float> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 75);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                     thrust::minus<float>());
 *  // V3 is now {-74, -75, -76, ..., -925}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/minus.html
 *  \see binary_function
 */
template<typename T>
  struct minus : public binary_function<T,T,T>
{
  /*! Function call operator. The return value is <tt>lhs - rhs</tt>.
   */
  __host__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs - rhs;}
}; // end minus

/*! \p multiplies is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>minus<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x*y</tt>.
 *
 *  \tparam T is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x*y</tt> must be defined and must have a return type that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>multiplies</tt> to multiply
 *  two device_vectors of \c floats.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *  thrust::device_vector<float> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 75);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                     thrust::multiplies<float>());
 *  // V3 is now {75, 150, 225, ..., 75000}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/multiplies.html
 *  \see binary_function
 */
template<typename T>
  struct multiplies : public binary_function<T,T,T>
{
  /*! Function call operator. The return value is <tt>lhs * rhs</tt>.
   */
  __host__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs * rhs;}
}; // end multiplies

/*! \p divides is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>divides<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x/y</tt>.
 *
 *  \tparam T is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x/y</tt> must be defined and must have a return type that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>divides</tt> to divide
 *  one device_vectors of \c floats by another.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *  thrust::device_vector<float> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 75);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                     thrust::divides<float>());
 *  // V3 is now {1/75, 2/75, 3/75, ..., 1000/75}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/divides.html
 *  \see binary_function
 */
template<typename T>
  struct divides : public binary_function<T,T,T>
{
  /*! Function call operator. The return value is <tt>lhs / rhs</tt>.
   */
  __host__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs / rhs;}
}; // end divides

/*! \p modulus is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>divides<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x%y</tt>.
 *
 *  \tparam T is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x%y</tt> must be defined and must have a return type that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>modulus</tt> to take
 *  the modulus of one device_vectors of \c floats by another.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *  thrust::device_vector<float> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 75);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                     thrust::modulus<int>());
 *  // V3 is now {1%75, 2%75, 3%75, ..., 1000%75}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/modulus.html
 *  \see binary_function
 */
template<typename T>
  struct modulus : public binary_function<T,T,T>
{
  /*! Function call operator. The return value is <tt>lhs % rhs</tt>.
   */
  __host__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs % rhs;}
}; // end modulus

/*! \p negate is a function object. Specifically, it is an Adaptable Unary Function.
 *  If \c f is an object of class <tt>negate<T></tt>, and \c x is an object
 *  of class \c T, then <tt>f(x)</tt> returns <tt>-x</tt>.
 *
 *  \tparam T is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>,
 *          and if \c x is an object of type \p T, then <tt>-x</tt> must be defined and must have a return type that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>negate</tt> to negate
 *  the element of a device_vector of \c floats.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(),
 *                     thrust::negate<float>());
 *  // V2 is now {-1, -2, -3, ..., -1000}
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/negate.html
 *  \see unary_function
 */
template<typename T>
  struct negate : public unary_function<T,T>
{
  /*! Function call operator. The return value is <tt>-x</tt>.
   */
  __host__ __device__ T operator()(const T &x) const {return -x;}
}; // end negate

/*! \}
 */

/*! \addtogroup comparison_operations Comparison Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

/*! \p equal_to is a function object. Specifically, it is an Adaptable Binary
 *  Predicate, which means it is a function object that tests the truth or falsehood
 *  of some condition. If \c f is an object of class <tt>equal_to<T></tt> and \c x
 *  and \c y are objects of class \c T, then <tt>f(x,y)</tt> returns \c true if
 *  <tt>x == y</tt> and \c false otherwise.
 *
 *  \tparam T is a model of <a href="http://www.sgi.com/tech/stl/EqualityComparable.html">Equality Comparable</a>.
 *
 *  \see http://www.sgi.com/tech/stl/equal_to.html
 *  \see binary_function
 */
template<typename T>
  struct equal_to : public binary_function<T,T,bool>
{
  /*! Function call operator. The return value is <tt>lhs == rhs</tt>.
   */
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {return lhs == rhs;}
}; // end equal_to

/*! \p not_equal_to is a function object. Specifically, it is an Adaptable Binary
 *  Predicate, which means it is a function object that tests the truth or falsehood
 *  of some condition. If \c f is an object of class <tt>not_equal_to<T></tt> and \c x
 *  and \c y are objects of class \c T, then <tt>f(x,y)</tt> returns \c true if
 *  <tt>x != y</tt> and \c false otherwise.
 *
 *  \tparam T is a model of <a href="http://www.sgi.com/tech/stl/EqualityComparable.html">Equality Comparable</a>.
 *
 *  \see http://www.sgi.com/tech/stl/not_equal_to.html
 *  \see binary_function
 */
template<typename T>
  struct not_equal_to : public binary_function<T,T,bool>
{
  /*! Function call operator. The return value is <tt>lhs != rhs</tt>.
   */
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {return lhs != rhs;}
}; // end not_equal_to

/*! \p greater is a function object. Specifically, it is an Adaptable Binary
 *  Predicate, which means it is a function object that tests the truth or falsehood
 *  of some condition. If \c f is an object of class <tt>greater<T></tt> and \c x
 *  and \c y are objects of class \c T, then <tt>f(x,y)</tt> returns \c true if
 *  <tt>x > y</tt> and \c false otherwise.
 *
 *  \tparam T is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable.html">LessThan Comparable</a>.
 *
 *  \see http://www.sgi.com/tech/stl/greater.html
 *  \see binary_function
 */
template<typename T>
  struct greater : public binary_function<T,T,bool>
{
  /*! Function call operator. The return value is <tt>lhs > rhs</tt>.
   */
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {return lhs > rhs;}
}; // end greater

/*! \p less is a function object. Specifically, it is an Adaptable Binary
 *  Predicate, which means it is a function object that tests the truth or falsehood
 *  of some condition. If \c f is an object of class <tt>less<T></tt> and \c x
 *  and \c y are objects of class \c T, then <tt>f(x,y)</tt> returns \c true if
 *  <tt>x < y</tt> and \c false otherwise.
 *
 *  \tparam T is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable.html">LessThan Comparable</a>.
 *
 *  \see http://www.sgi.com/tech/stl/less.html
 *  \see binary_function
 */
template<typename T>
  struct less : public binary_function<T,T,bool>
{
  /*! Function call operator. The return value is <tt>lhs < rhs</tt>.
   */
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {return lhs < rhs;}
}; // end less

/*! \p greater_equal is a function object. Specifically, it is an Adaptable Binary
 *  Predicate, which means it is a function object that tests the truth or falsehood
 *  of some condition. If \c f is an object of class <tt>greater_equal<T></tt> and \c x
 *  and \c y are objects of class \c T, then <tt>f(x,y)</tt> returns \c true if
 *  <tt>x >= y</tt> and \c false otherwise.
 *
 *  \tparam T is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable.html">LessThan Comparable</a>.
 *
 *  \see http://www.sgi.com/tech/stl/greater_equal.html
 *  \see binary_function
 */
template<typename T>
  struct greater_equal : public binary_function<T,T,bool>
{
  /*! Function call operator. The return value is <tt>lhs >= rhs</tt>.
   */
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {return lhs >= rhs;}
}; // end greater_equal

/*! \p less_equal is a function object. Specifically, it is an Adaptable Binary
 *  Predicate, which means it is a function object that tests the truth or falsehood
 *  of some condition. If \c f is an object of class <tt>less_equal<T></tt> and \c x
 *  and \c y are objects of class \c T, then <tt>f(x,y)</tt> returns \c true if
 *  <tt>x <= y</tt> and \c false otherwise.
 *
 *  \tparam T is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable.html">LessThan Comparable</a>.
 *
 *  \see http://www.sgi.com/tech/stl/less_equal.html
 *  \see binary_function
 */
template<typename T>
  struct less_equal : public binary_function<T,T,bool>
{
  /*! Function call operator. The return value is <tt>lhs <= rhs</tt>.
   */
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {return lhs <= rhs;}
}; // end less_equal

/*! \}
 */


/*! \addtogroup logical_operations Logical Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

/*! \p logical_and is a function object. Specifically, it is an Adaptable Binary Predicate,
 *  which means it is a function object that tests the truth or falsehood of some condition.
 *  If \c f is an object of class <tt>logical_and<T></tt> and \c x and \c y are objects of
 *  class \c T (where \c T is convertible to \c bool) then <tt>f(x,y)</tt> returns \c true
 *  if and only if both \c x and \c y are \c true.
 *
 *  \tparam T must be convertible to \c bool.
 *
 *  \see http://www.sgi.com/tech/stl/logical_and.html
 *  \see binary_function
 */
template<typename T>
  struct logical_and : public binary_function<T,T,bool>
{
  /*! Function call operator. The return value is <tt>lhs && rhs</tt>.
   */
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {return lhs && rhs;}
}; // end logical_and

/*! \p logical_or is a function object. Specifically, it is an Adaptable Binary Predicate,
 *  which means it is a function object that tests the truth or falsehood of some condition.
 *  If \c f is an object of class <tt>logical_or<T></tt> and \c x and \c y are objects of
 *  class \c T (where \c T is convertible to \c bool) then <tt>f(x,y)</tt> returns \c true
 *  if and only if either \c x or \c y are \c true.
 *
 *  \tparam T must be convertible to \c bool.
 *
 *  \see http://www.sgi.com/tech/stl/logical_or.html
 *  \see binary_function
 */
template<typename T>
  struct logical_or : public binary_function<T,T,bool>
{
  /*! Function call operator. The return value is <tt>lhs || rhs</tt>.
   */
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {return lhs || rhs;}
}; // end logical_or

/*! \p logical_not is a function object. Specifically, it is an Adaptable Predicate,
 *  which means it is a function object that tests the truth or falsehood of some condition.
 *  If \c f is an object of class <tt>logical_not<T></tt> and \c x is an object of
 *  class \c T (where \c T is convertible to \c bool) then <tt>f(x)</tt> returns \c true
 *  if and only if \c x is \c false.
 *
 *  \tparam T must be convertible to \c bool.
 *
 *  The following code snippet demonstrates how to use \p logical_not to transform
 *  a device_vector of \c bools into its logical complement.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/transform.h>
 *  #include <thrust/functional.h>
 *  ...
 *  thrust::device_vector<bool> V;
 *  ...
 *  thrust::transform(V.begin(), V.end(), V.begin(), thrust::logical_not<bool>());
 *  // The elements of V are now the logical complement of what they were prior
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/logical_not.html
 *  \see unary_function
 */
template<typename T>
  struct logical_not : public unary_function<T,bool>
{
  /*! Function call operator. The return value is <tt>!x</tt>.
   */
  __host__ __device__ bool operator()(const T &x) const {return !x;}
}; // end logical_not

/*! \}
 */

/*! \addtogroup bitwise_operations Bitwise Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

/*! \p bit_and is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>bit_and<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x&y</tt>.
 *
 *  \tparam T is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x&y</tt> must be defined and must have a return type that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>bit_and</tt> to take
 *  the bitwise AND of one device_vector of \c ints by another.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<int> V1(N);
 *  thrust::device_vector<int> V2(N);
 *  thrust::device_vector<int> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 13);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::bit_and<int>());
 *  // V3 is now {1&13, 2&13, 3&13, ..., 1000%13}
 *  \endcode
 *
 *  \see binary_function
 */
template<typename T>
  struct bit_and : public binary_function<T,T,T>
{
  /*! Function call operator. The return value is <tt>lhs & rhs</tt>.
   */
  __host__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs & rhs;}
}; // end bit_and

/*! \p bit_or is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>bit_and<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x|y</tt>.
 *
 *  \tparam T is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x|y</tt> must be defined and must have a return type that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>bit_or</tt> to take
 *  the bitwise OR of one device_vector of \c ints by another.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<int> V1(N);
 *  thrust::device_vector<int> V2(N);
 *  thrust::device_vector<int> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 13);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::bit_or<int>());
 *  // V3 is now {1|13, 2|13, 3|13, ..., 1000|13}
 *  \endcode
 *
 *  \see binary_function
 */
template<typename T>
  struct bit_or : public binary_function<T,T,T>
{
  /*! Function call operator. The return value is <tt>lhs | rhs</tt>.
   */
  __host__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs | rhs;}
}; // end bit_or

/*! \p bit_xor is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>bit_and<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x^y</tt>.
 *
 *  \tparam T is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x^y</tt> must be defined and must have a return type that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>bit_xor</tt> to take
 *  the bitwise XOR of one device_vector of \c ints by another.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<int> V1(N);
 *  thrust::device_vector<int> V2(N);
 *  thrust::device_vector<int> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 13);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::bit_xor<int>());
 *  // V3 is now {1^13, 2^13, 3^13, ..., 1000^13}
 *  \endcode
 *
 *  \see binary_function
 */
template<typename T>
  struct bit_xor : public binary_function<T,T,T>
{
  /*! Function call operator. The return value is <tt>lhs ^ rhs</tt>.
   */
  __host__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs ^ rhs;}
}; // end bit_xor

/*! \}
 */

/*! \addtogroup generalized_identity_operations Generalized Identity Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

/*! \p identity is a Unary Function that represents the identity function: it takes
 *  a single argument \c x, and returns \c x.
 *
 *  \tparam T No requirements on \p T.
 *
 *  The following code snippet demonstrates that \p identity returns its
 *  argument.
 *
 *  \code
 *  #include <thrust/functional.h>
 *  #include <assert.h>
 *  ...
 *  int x = 137;
 *  thrust::identity<int> id;
 *  assert(x == id(x));
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/identity.html
 *  \see unary_function
 */
template<typename T>
  struct identity : public unary_function<T,T>
{
  /*! Function call operator. The return value is <tt>x</tt>.
   */
  __host__ __device__ const T &operator()(const T &x) const {return x;}
}; // end identity

/*! \p maximum is a function object that takes two arguments and returns the greater
 *  of the two. Specifically, it is an Adaptable Binary Function. If \c f is an
 *  object of class <tt>maximum<T></tt> and \c x and \c y are objects of class \c T
 *  <tt>f(x,y)</tt> returns \c x if <tt>x > y</tt> and \c y, otherwise.
 *
 *  \tparam T is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable.html">LessThan Comparable</a>.
 *
 *  The following code snippet demonstrates that \p maximum returns its
 *  greater argument.
 *
 *  \code
 *  #include <thrust/functional.h>
 *  #include <assert.h>
 *  ...
 *  int x =  137;
 *  int y = -137;
 *  thrust::maximum<int> mx;
 *  assert(x == mx(x,y));
 *  \endcode
 *
 *  \see minimum
 *  \see min
 *  \see binary_function
 */
template<typename T>
  struct maximum : public binary_function<T,T,T>
{
  /*! Function call operator. The return value is <tt>rhs < lhs ? lhs : rhs</tt>.
   */
  __host__ __device__ const T operator()(const T &lhs, const T &rhs) const {return lhs < rhs ? rhs : lhs;}
}; // end maximum

/*! \p minimum is a function object that takes two arguments and returns the lesser
 *  of the two. Specifically, it is an Adaptable Binary Function. If \c f is an
 *  object of class <tt>minimum<T></tt> and \c x and \c y are objects of class \c T
 *  <tt>f(x,y)</tt> returns \c x if <tt>x < y</tt> and \c y, otherwise.
 *
 *  \tparam T is a model of <a href="http://www.sgi.com/tech/stl/LessThanComparable.html">LessThan Comparable</a>.
 *
 *  The following code snippet demonstrates that \p minimum returns its
 *  lesser argument.
 *
 *  \code
 *  #include <thrust/functional.h>
 *  #include <assert.h>
 *  ...
 *  int x =  137;
 *  int y = -137;
 *  thrust::minimum<int> mn;
 *  assert(y == mn(x,y));
 *  \endcode
 *
 *  \see maximum
 *  \see max
 *  \see binary_function
 */
template<typename T>
  struct minimum : public binary_function<T,T,T>
{
  /*! Function call operator. The return value is <tt>lhs < rhs ? lhs : rhs</tt>.
   */
  __host__ __device__ const T operator()(const T &lhs, const T &rhs) const {return lhs < rhs ? lhs : rhs;}
}; // end minimum

/*! \p project1st is a function object that takes two arguments and returns 
 *  its first argument; the second argument is unused. It is essentially a
 *  generalization of identity to the case of a Binary Function.
 *
 *  \code
 *  #include <thrust/functional.h>
 *  #include <assert.h>
 *  ...
 *  int x =  137;
 *  int y = -137;
 *  thrust::project1st<int> pj1;
 *  assert(x == pj1(x,y));
 *  \endcode
 *
 *  \see identity
 *  \see project2nd
 *  \see binary_function
 */
template<typename T1, typename T2>
  struct project1st : public binary_function<T1,T2,T1>
{
  /*! Function call operator. The return value is <tt>lhs</tt>.
   */
  __host__ __device__ const T1 &operator()(const T1 &lhs, const T2 &rhs) const {return lhs;}
}; // end project1st

/*! \p project2nd is a function object that takes two arguments and returns 
 *  its second argument; the first argument is unused. It is essentially a
 *  generalization of identity to the case of a Binary Function.
 *
 *  \code
 *  #include <thrust/functional.h>
 *  #include <assert.h>
 *  ...
 *  int x =  137;
 *  int y = -137;
 *  thrust::project2nd<int> pj2;
 *  assert(y == pj2(x,y));
 *  \endcode
 *
 *  \see identity
 *  \see project1st
 *  \see binary_function
 */
template<typename T1, typename T2>
  struct project2nd : public binary_function<T1,T2,T2>
{
  /*! Function call operator. The return value is <tt>rhs</tt>.
   */
  __host__ __device__ const T2 &operator()(const T1 &lhs, const T2 &rhs) const {return rhs;}
}; // end project2nd

/*! \}
 */


// odds and ends

/*! \addtogroup function_object_adaptors
 *  \{
 */

/*! \p unary_negate is a function object adaptor: it is an Adaptable Predicate
 *  that represents the logical negation of some other Adaptable Predicate.
 *  That is: if \c f is an object of class <tt>unary_negate<AdaptablePredicate></tt>,
 *  then there exists an object \c pred of class \c AdaptablePredicate such
 *  that <tt>f(x)</tt> always returns the same value as <tt>!pred(x)</tt>.
 *  There is rarely any reason to construct a <tt>unary_negate</tt> directly;
 *  it is almost always easier to use the helper function not1.
 *
 *  \see http://www.sgi.com/tech/stl/unary_negate.html
 *  \see not1
 */
template<typename Predicate>
struct unary_negate 
    : public thrust::unary_function<typename Predicate::argument_type, bool>
{
  __host__ __device__
  explicit unary_negate(Predicate p) : pred(p){}

  /*! Function call operator. The return value is <tt>!pred(x)</tt>.
   */
  __host__ __device__
  bool operator()(const typename Predicate::argument_type& x) { return !pred(x); }

  Predicate pred;
}; // end unary_negate

/*! \p not1 is a helper function to simplify the creation of Adaptable Predicates:
 *  it takes an Adaptable Predicate \p pred as an argument and returns a new Adaptable
 *  Predicate that represents the negation of \p pred. That is: if \c pred is an object
 *  of a type which models Adaptable Predicate, then the the type of the result
 *  \c npred of <tt>not1(pred)</tt> is also a model of Adaptable Predicate and
 *  <tt>npred(x)</tt> always returns the same value as <tt>!pred(x)</tt>.
 *
 *  \param pred The Adaptable Predicate to negate.
 *  \return A new object, <tt>npred</tt> such that <tt>npred(x)</tt> always returns
 *          the same value as <tt>!pred(x)</tt>.
 *
 *  \tparam Predicate is a model of <a href="http://www.sgi.com/tech/stl/AdaptablePredicate.html">Adaptable Predicate</a>.
 *
 *  \see unary_negate
 *  \see not2
 */
template<typename Predicate>
  __host__ __device__
  unary_negate<Predicate> not1(const Predicate &pred);

/*! \p binary_negate is a function object adaptor: it is an Adaptable Binary 
 *  Predicate that represents the logical negation of some other Adaptable
 *  Binary Predicate. That is: if \c f is an object of class <tt>binary_negate<AdaptablePredicate></tt>,
 *  then there exists an object \c pred of class \c AdaptableBinaryPredicate
 *  such that <tt>f(x,y)</tt> always returns the same value as <tt>!pred(x,y)</tt>.
 *  There is rarely any reason to construct a <tt>binary_negate</tt> directly;
 *  it is almost always easier to use the helper function not2.
 *
 *  \see http://www.sgi.com/tech/stl/binary_negate.html
 */
template<typename Predicate>
struct binary_negate
    : public thrust::binary_function<typename Predicate::first_argument_type,
                                     typename Predicate::second_argument_type,
                                     bool>
{
  __host__ __device__
  explicit binary_negate(Predicate p) : pred(p){}

  /*! Function call operator. The return value is <tt>!pred(x,y)</tt>.
   */
  __host__ __device__
  bool operator()(const typename Predicate::first_argument_type& x, const typename Predicate::second_argument_type& y)
  { 
      return !pred(x,y); 
  }

  Predicate pred;
}; // end binary_negate

/*! \p not2 is a helper function to simplify the creation of Adaptable Binary Predicates:
 *  it takes an Adaptable Binary Predicate \p pred as an argument and returns a new Adaptable
 *  Binary Predicate that represents the negation of \p pred. That is: if \c pred is an object
 *  of a type which models Adaptable Binary Predicate, then the the type of the result
 *  \c npred of <tt>not2(pred)</tt> is also a model of Adaptable Binary Predicate and
 *  <tt>npred(x,y)</tt> always returns the same value as <tt>!pred(x,y)</tt>.
 *
 *  \param pred The Adaptable Binary Predicate to negate.
 *  \return A new object, <tt>npred</tt> such that <tt>npred(x,y)</tt> always returns
 *          the same value as <tt>!pred(x,y)</tt>.
 *
 *  \tparam Binary Predicate is a model of <a href="http://www.sgi.com/tech/stl/AdaptableBinaryPredicate.html">Adaptable Binary Predicate</a>.
 *
 *  \see binary_negate
 *  \see not1
 */
template<typename BinaryPredicate>
  __host__ __device__
  binary_negate<BinaryPredicate> not2(const BinaryPredicate &pred);

/*! \}
 */

namespace placeholders
{

static const thrust::detail::functional::placeholder<0>::type _1;
static const thrust::detail::functional::placeholder<1>::type _2;
static const thrust::detail::functional::placeholder<2>::type _3;
static const thrust::detail::functional::placeholder<3>::type _4;
static const thrust::detail::functional::placeholder<4>::type _5;
static const thrust::detail::functional::placeholder<5>::type _6;
static const thrust::detail::functional::placeholder<6>::type _7;
static const thrust::detail::functional::placeholder<7>::type _8;
static const thrust::detail::functional::placeholder<8>::type _9;
static const thrust::detail::functional::placeholder<9>::type _10;

} // end placeholders

} // end thrust

#include <thrust/detail/functional.inl>
#include <thrust/detail/functional/operators.h>

