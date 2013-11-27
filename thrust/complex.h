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

/*
 * Copyright (c) 2010, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from U.S. Dept. of Energy) All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the
 * following conditions are met:
 * 
 *     * Redistributions of source code must retain the above
 * copyright notice, this list of conditions and the following
 * disclaimer.
 * 
 *     * Redistributions in binary form must reproduce the
 * above copyright notice, this list of conditions and the
 * following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 * 
 *     * Neither the name of the University of California,
 * Berkeley, nor the names of its contributors may be used to
 * endorse or promote products derived from this software
 * without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * 
*/

/*! \file complex.h
 *  \brief Complex numbers
 */

#pragma once

#include <thrust/detail/config.h>

#if (1 || defined THRUST_DEVICE_BACKEND && THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_CUDA) || (defined THRUST_DEVICE_SYSTEM && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA)

#ifdef _WIN32
#define _USE_MATH_DEFINES 1  // make sure M_PI is defined
#endif

#include <math.h>
#include <complex>
#include <sstream>
#include <thrust/cmath.h>
#include <thrust/detail/type_traits.h>
#include <cfloat>


namespace thrust
{

/*! \addtogroup numerics
 *  \{
 */

/*! \addtogroup complex_numbers Complex Numbers
 *  \{
 */

  /*! \p complex is the Thrust equivalent to std::complex. It is functionally
   *  equivalent to it, but can also be used in device code which std::complex currently cannot.
   *
   *  \tparam T The type used to hold the real and imaginary parts. Should be <tt>float</tt> 
   *  or <tt>double</tt>. Others types are not supported.
   *
   */
template <typename T>
struct complex
{
public:

  typedef T value_type;



  /* --- Constructors --- */

  /*! Construct a complex number from its real and imaginary parts.
   *
   *  \param re The real part of the number.
   *  \param im The imaginary part of the number.
   */
  inline __host__ __device__      
  complex(const T & re = T(), const T& im = T())
  {
    real(re);
    imag(im);
  }  

  /*! This copy constructor copies from a \p complex with a type that
   *  is convertible to this \p complex \c value_type.
   *
   *  \param z The \p complex to copy from.
   *
   *  \tparam X is convertible to \c value_type.
   */
  template <typename X> 
  inline __host__ __device__
  complex<T>(const complex<X> & z)
  {
    real(z.real());
    imag(z.imag());
  }  
  
  /*! This copy constructor copies from a <tt>std::complex</tt> with a type that
   *  is convertible to this \p complex \c value_type.
   *
   *  \param z The \p complex to copy from.
   *
   *  \tparam X is convertible to \c value_type.
   */
  template <typename X> 
    __host__
    inline complex<T>(const std::complex<X> & z)
    {
      real(z.real());
      imag(z.imag());
    }  



  /* --- Compound Assignment Operators --- */

  /*! Adds a \p complex to this \p complex and 
   *  assigns the result to this \p complex.
   *
   *  \param z The \p complex to be Added.
   */
  __host__ __device__
    inline complex<T>& operator+=(const complex<T> z)
    {
      real(real()+z.real());
      imag(imag()+z.imag());
      return *this;
    }

  /*! Subtracts a \p complex from this \p complex and 
   *  assigns the result to this \p complex.
   *
   *  \param z The \p complex to be subtracted.
   */
  __host__ __device__
    inline complex<T>& operator-=(const complex<T> z)
    {
      real(real()-z.real());
      imag(imag()-z.imag());
      return *this;
    }

  /*! Multiplies this \p complex by another \p complex and 
   *  assigns the result to this \p complex.
   *
   *  \param z The \p complex to be multiplied.
   */
  __host__ __device__
    inline complex<T>& operator*=(const complex<T> z)
    {
      *this = *this * z;
      return *this;
    }

  /*! Divides this \p complex by another \p complex and 
   *  assigns the result to this \p complex.
   *
   *  \param z The \p complex to be divided.
   */
  __host__ __device__
    inline complex<T>& operator/=(const complex<T> z)
    {
      *this = *this / z;
      return *this;
    }



  /* --- Getter functions --- 
   * The volatile ones are there to help for example
   * with certain reductions optimizations
   */

  /*! Returns the real part of this \p complex.
   */
  __host__ __device__ inline T real() const volatile{ return _m[0]; }

  /*! Returns the imaginary part of this \p complex.
   */
  __host__ __device__ inline T imag() const volatile{ return _m[1]; }

  /*! Returns the real part of this \p complex.
   */
  __host__ __device__ inline T real() const{ return _m[0]; }

  /*! Returns the imaginary part of this \p complex.
   */
  __host__ __device__ inline T imag() const{ return _m[1]; }



  /* --- Setter functions --- 
   * The volatile ones are there to help for example
   * with certain reductions optimizations
   */

  /*! Sets the real part of this \p complex.
   *
   *  \param re The new real part of this \p complex.
   */
  __host__ __device__ inline void real(T re)volatile{ _m[0] = re; }

  /*! Sets the imaginary part of this \p complex.
   *
   *  \param im The new imaginary part of this \p complex.e
   */
  __host__ __device__ inline void imag(T im)volatile{ _m[1] = im; }

  /*! Sets the real part of this \p complex.
   *
   *  \param re The new real part of this \p complex.
   */
  __host__ __device__ inline void real(T re){ _m[0] = re; }

  /*! Sets the imaginary part of this \p complex.
   *
   *  \param im The new imaginary part of this \p complex.
   */
  __host__ __device__ inline void imag(T im){ _m[1] = im; }



  /* --- Casting functions --- */

  /*! Casts this \p complex to a <tt>std::complex</tt> of the same type.
   */
  inline operator std::complex<T>() const { return std::complex<T>(real(),imag()); }

private:
  T _m[2];
};



  namespace detail{
    template<typename T1, typename T2, typename Enable = void> struct promoted_numerical_type;

    template<typename T1, typename T2> struct promoted_numerical_type<T1,T2,typename enable_if<and_<
      typename is_floating_point<T1>::type,
      typename is_floating_point<T2>::type>::value
      >::type>{
	typedef larger_type<T1,T2> type;
    };

    template<typename T1, typename T2> struct promoted_numerical_type<T1,T2,typename enable_if<and_<
      typename is_integral<T1>::type,
      typename is_floating_point<T2>::type>::value
      >::type>{
      typedef T2 type;
    };
    template<typename T1, typename T2> struct promoted_numerical_type<T1,T2,typename enable_if<and_<
      typename is_floating_point<T1>::type,
      typename is_integral<T2>::type>::value
      >::type>{
      typedef T1 type;
    };

  }
  


  /* --- General Functions --- */

  /*! Returns the magnitude (also known as absolute value) of a \p complex.
   *
   *  \param z The \p complex from which to calculate the absolute value.
   */
  template<typename T> __host__ __device__ T abs(const complex<T>& z);

  /*! Returns the phase angle (also known as argument) in radians of a \p complex.
   *
   *  \param z The \p complex from which to calculate the phase angle.
   */
  template<typename T> __host__ __device__ T arg(const complex<T>& z);

  /*! Returns the square of the magnitude of a \p complex.
   *
   *  \param z The \p complex from which to calculate the norm.
   */
  template<typename T> __host__ __device__ T norm(const complex<T>& z);

  /*! Returns the complex conjugate of a \p complex.
   *
   *  \param z The \p complex from which to calculate the complex conjugate.
   */
  template<typename T> __host__ __device__ complex<T> conj(const complex<T>& z);

  /*! Returns a \p complex with the specified magnitude and phase.
   *
   *  \param m The magnitude of the returned \p complex.
   *  \param theta The phase of the returned \p complex in radians.
   */
  template<typename T> __host__ __device__ complex<T> polar(const T& m, const T& theta = 0);



  /* --- Binary Arithmetic operators --- */

  /*! Multiplies two \p complex numbers.
   *
   *  \param lhs The first \p complex.
   *  \param rhs The second \p complex.
   */
  template <typename T> __host__ __device__ inline complex<T> operator*(const complex<T>& lhs, const complex<T>& rhs);

  /*! Multiplies a \p complex number by a scalar.
   *
   *  \param lhs The \p complex.
   *  \param rhs The scalar.
   */
  template <typename T> __host__ __device__ inline complex<T> operator*(const complex<T>& lhs, const T & rhs);

  /*! Multiplies a scalr by a \p complex number.
   *
   *  \param lhs The scalar.
   *  \param rhs The \p complex.
   */
  template <typename T> __host__ __device__ inline complex<T> operator*(const T& lhs, const complex<T>& rhs);

  /*! Divides two \p complex numbers.
   *
   *  \param lhs The numerator (dividend).
   *  \param rhs The denomimator (divisor).
   */
  template <typename T> __host__ __device__ inline complex<T> operator/(const complex<T>& lhs, const complex<T>& rhs);

  /*! Divides a \p complex number by a scalar.
   *
   *  \param lhs The complex numerator (dividend).
   *  \param rhs The scalar denomimator (divisor).
   */
  template <typename T> __host__ __device__ inline complex<T> operator/(const complex<T>& lhs, const T & rhs);

  /*! Divides a scalar by a \p complex number.
   *
   *  \param lhs The scalar numerator (dividend).
   *  \param rhs The complex denomimator (divisor).
   */
  template <typename T> __host__ __device__ inline complex<T> operator/(const T& lhs, const complex<T> & rhs);

  /*! Adds two \p complex numbers.
   *
   *  \param lhs The first \p complex.
   *  \param rhs The second \p complex.
   */
  template <typename T> __host__ __device__ inline complex<T> operator+(const complex<T>& lhs, const complex<T>& rhs);

  /*! Adds a scalar to a \p complex number.
   *
   *  \param lhs The \p complex.
   *  \param rhs The scalar.
   */
  template <typename T> __host__ __device__ inline complex<T> operator+(const complex<T>& lhs, const T & rhs);

  /*! Adds a \p complex number to a scalar.
   *
   *  \param lhs The scalar.
   *  \param rhs The \p complex.
   */
  template <typename T> __host__ __device__ inline complex<T> operator+(const T& lhs, const complex<T>& rhs);

  /*! Subtracts two \p complex numbers.
   *
   *  \param lhs The first \p complex (minuend).
   *  \param rhs The second \p complex (subtrahend).
   */
  template <typename T> __host__ __device__ inline complex<T> operator-(const complex<T>& lhs, const complex<T>& rhs);

  /*! Subtracts a scalar from a \p complex number.
   *
   *  \param lhs The \p complex (minuend).
   *  \param rhs The scalar (subtrahend).
   */
  template <typename T> __host__ __device__ inline complex<T> operator-(const complex<T>& lhs, const T & rhs);

  /*! Subtracts a \p complex number from a scalar.
   *
   *  \param lhs The scalar (minuend).
   *  \param rhs The \p complex (subtrahend).
   */
  template <typename T> __host__ __device__ inline complex<T> operator-(const T& lhs, const complex<T>& rhs);



  /* --- Unary Arithmetic operators --- */

  /*! Unary plus, returns its \p complex argument.
   *
   *  \param rhs The \p complex argument.
   */
  template <typename T> __host__ __device__ inline complex<T> operator+(const complex<T>& rhs);

  /*! Unary minus, returns the additive inverse (negation) of its \p complex argument.
   *
   *  \param rhs The \p complex argument.
   */
  template <typename T> __host__ __device__ inline complex<T> operator-(const complex<T>& rhs);



  /* --- Exponential Functions --- */

  /*! Returns the complex exponential of a \p complex number.
   *
   *  \param rhs The \p complex argument.
   */
  template <typename T> __host__ __device__ complex<T> exp(const complex<T>& z);

  /*! Returns the complex natural logarithm of a \p complex number.
   *
   *  \param rhs The \p complex argument.
   */
  template <typename T> __host__ __device__ complex<T> log(const complex<T>& z);

  /*! Returns the complex base 10 logarithm of a \p complex number.
   *
   *  \param rhs The \p complex argument.
   */
  template <typename T> __host__ __device__ complex<T> log10(const complex<T>& z);



  /* --- Power Functions --- */

  /*! Returns a \p complex number raised to another.
   *
   *  \param x The base.
   *  \param y The exponent.
   */
  template <typename T> __host__ __device__ complex<T> pow(const complex<T>& x, const complex<T>& y);

  /*! Returns a \p complex number raised to a scalar.
   *
   *  \param x The \p complex base.
   *  \param y The scalar exponent.
   */
  template <typename T> __host__ __device__ complex<T> pow(const complex<T>& x, const T& y);

  /*! Returns a scalar raised to a \p complex number.
   *
   *  \param x The scalar base.
   *  \param y The \p complex exponent.
   */
  template <typename T> __host__ __device__ complex<T> pow(const T& x, const complex<T>& y);

  /*! Returns a \p complex number raised to another. The types of the two \p complex should be compatible
   * and the type of the returned \p complex is the promoted type of the two arguments.
   *
   *  \param x The base.
   *  \param y The exponent.
   */
  template <typename T, typename U> __host__ __device__ complex<typename detail::promoted_numerical_type<T,U>::type > pow(const complex<T>& x, const complex<U>& y);

  /*! Returns a \p complex number raised to a scalar. The type of the \p complex should be compatible with the scalar
   * and the type of the returned \p complex is the promoted type of the two arguments.
   *
   *  \param x The base.
   *  \param y The exponent.
   */
  template <typename T, typename U> __host__ __device__ complex<typename detail::promoted_numerical_type<T,U>::type > pow(const complex<T>& x, const U& y);

  /*! Returns a scalar raised to a \p complex number. The type of the \p complex should be compatible with the scalar
   * and the type of the returned \p complex is the promoted type of the two arguments.
   *
   *  \param x The base.
   *  \param y The exponent.
   */
  template <typename T, typename U> __host__ __device__ complex<typename detail::promoted_numerical_type<T,U>::type > pow(const T& x,const complex<U>& y);

  /*! Returns the complex square root of a \p complex number.
   *
   *  \param x The \p complex argument.
   */
  template <typename T> __host__ __device__ complex<T> sqrt(const complex<T>&x);



  /* --- Trigonometric Functions --- */

  /*! Returns the complex cosine of a \p complex number.
   *
   *  \param x The \p complex argument.
   */
  template <typename T> __host__ __device__ complex<T> cos(const complex<T>&x);

  /*! Returns the complex sine of a \p complex number.
   *
   *  \param x The \p complex argument.
   */
  template <typename T> __host__ __device__ complex<T> sin(const complex<T>&x);

  /*! Returns the complex tangent of a \p complex number.
   *
   *  \param x The \p complex argument.
   */
  template <typename T> __host__ __device__ complex<T> tan(const complex<T>&x);



  /* --- Hyperbolic Functions --- */

  /*! Returns the complex hyperbolic cosine of a \p complex number.
   *
   *  \param x The \p complex argument.
   */
  template <typename T> __host__ __device__ complex<T> cosh(const complex<T>& x);

  /*! Returns the complex hyperbolic sine of a \p complex number.
   *
   *  \param x The \p complex argument.
   */
  template <typename T> __host__ __device__ complex<T> sinh(const complex<T>&x);

  /*! Returns the complex hyperbolic tangent of a \p complex number.
   *
   *  \param x The \p complex argument.
   */
  template <typename T> __host__ __device__ complex<T> tanh(const complex<T>&x);



  /* --- Inverse Trigonometric Functions --- */

  /*! Returns the complex arc cosine of a \p complex number.
   *
   *  The range of the real part of the result is [0, Pi] and 
   *  the range of the imaginary part is [-inf, +inf]
   *
   *  \param x The \p complex argument.
   */
  template <typename T> __host__ __device__ complex<T> acos(const complex<T>& x);

  /*! Returns the complex arc sine of a \p complex number.
   *
   *  The range of the real part of the result is [-Pi/2, Pi/2] and 
   *  the range of the imaginary part is [-inf, +inf]
   *
   *  \param x The \p complex argument.
   */
  template <typename T> __host__ __device__ complex<T> asin(const complex<T>& x);

  /*! Returns the complex arc tangent of a \p complex number.
   *
   *  The range of the real part of the result is [-Pi/2, Pi/2] and 
   *  the range of the imaginary part is [-inf, +inf]
   *
   *  \param x The \p complex argument.
   */
  template <typename T> __host__ __device__ complex<T> atan(const complex<T>& x);



  /* --- Inverse Hyperbolic Functions --- */

  /*! Returns the complex inverse hyperbolic cosine of a \p complex number.
   *
   *  The range of the real part of the result is [0, +inf] and 
   *  the range of the imaginary part is [-Pi, Pi]
   *
   *  \param x The \p complex argument.
   */
  template <typename T> __host__ __device__ complex<T> acosh(const complex<T>& x);

  /*! Returns the complex inverse hyperbolic sine of a \p complex number.
   *
   *  The range of the real part of the result is [-inf, +inf] and 
   *  the range of the imaginary part is [-Pi/2, Pi/2]
   *
   *  \param x The \p complex argument.
   */
  template <typename T> __host__ __device__ complex<T> asinh(const complex<T>& x);

  /*! Returns the complex inverse hyperbolic tangent of a \p complex number.
   *
   *  The range of the real part of the result is [-inf, +inf] and 
   *  the range of the imaginary part is [-Pi/2, Pi/2]
   *
   *  \param x The \p complex argument.
   */
  template <typename T> __host__ __device__ complex<T> atanh(const complex<T>& x);



  /* --- Stream Operators --- */

  /*! Writes to an output stream a \p complex number in the form (real,imaginary).
   *
   *  \param os The output stream.
   *  \param z The \p complex number to output.
   */
  template<typename ValueType,class charT, class traits>
    std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& os, const complex<ValueType>& z);

  /*! Reads a \p complex number from an input stream.
   *  The recognized formats are:
   * - real
   * - (real)
   * - (real, imaginary)
   *
   * The values read must be convertible to the \p complex's \c value_type 
   *
   *  \param is The input stream.
   *  \param z The \p complex number to set.
   */
  template<typename ValueType, typename charT, class traits>
    std::basic_istream<charT, traits>&
    operator>>(std::basic_istream<charT, traits>& is, complex<ValueType>& z);
  



  // Equality operators 
  template <typename ValueType> 
    __host__ __device__
    inline bool operator==(const complex<ValueType>& lhs, const complex<ValueType>& rhs){
    if(lhs.real() == rhs.real() && lhs.imag() == rhs.imag()){
      return true;
    }
    return false;
  }
  template <typename ValueType> 
    __host__ __device__
    inline bool operator==(const ValueType & lhs, const complex<ValueType>& rhs){
    if(lhs == rhs.real() && rhs.imag() == 0){
      return true;
    }
    return false;
  }
  template <typename ValueType> 
    __host__ __device__
    inline bool operator==(const complex<ValueType> & lhs, const ValueType& rhs){
    if(lhs.real() == rhs && lhs.imag() == 0){
      return true;
    }
    return false;
  }

  template <typename ValueType> 
    __host__ __device__
    inline bool operator!=(const complex<ValueType>& lhs, const complex<ValueType>& rhs){
    return !(lhs == rhs);
  }

  template <typename ValueType> 
    __host__ __device__
    inline bool operator!=(const ValueType & lhs, const complex<ValueType>& rhs){
    return !(lhs == rhs);
  }

  template <typename ValueType> 
    __host__ __device__
    inline bool operator!=(const complex<ValueType> & lhs, const ValueType& rhs){
    return !(lhs == rhs);
  }




  // Transcendental functions implementation


  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> log10(const complex<ValueType>& z){ 
    // Using the explicit literal prevents compile time warnings in
    // devices that don't support doubles 
    return thrust::log(z)/ValueType(2.30258509299404568402);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> pow(const complex<ValueType>& z, const ValueType & exponent){
    return thrust::exp(thrust::log(z)*exponent);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> pow(const complex<ValueType>& z, const complex<ValueType> & exponent){
    return thrust::exp(thrust::log(z)*exponent);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> pow(const ValueType & x, const complex<ValueType> & exponent){
    return thrust::exp(std::log(x)*exponent);
  }

  template <typename T, typename U>
    __host__ __device__ 
    inline complex<typename detail::promoted_numerical_type<T,U>::type > pow(const complex<T>& z, const complex<T>& exponent){
    typedef typename detail::promoted_numerical_type<T,U>::type PromotedType;
    return thrust::exp(thrust::log(complex<PromotedType>(z))*complex<PromotedType>(exponent));
  }

  template <typename T, typename U>
    __host__ __device__ 
    inline complex<typename detail::promoted_numerical_type<T,U>::type > pow(const complex<T>& z, const U& exponent){
    typedef typename detail::promoted_numerical_type<T,U>::type PromotedType;
    return thrust::exp(thrust::log(complex<PromotedType>(z))*PromotedType(exponent));
  }

  template <typename T, typename U>
    __host__ __device__ 
    inline complex<typename detail::promoted_numerical_type<T,U>::type > pow(const T& x, const complex<U>& exponent){
    typedef typename detail::promoted_numerical_type<T,U>::type PromotedType;
    return thrust::exp(std::log(PromotedType(x))*complex<PromotedType>(exponent));
  }


} // end namespace thrust

#include <thrust/detail/complex/cexp.h>
#include <thrust/detail/complex/cexpf.h>
#include <thrust/detail/complex/clog.h>
#include <thrust/detail/complex/clogf.h>
#include <thrust/detail/complex/ccosh.h>
#include <thrust/detail/complex/ccoshf.h>
#include <thrust/detail/complex/csinh.h>
#include <thrust/detail/complex/csinhf.h>
#include <thrust/detail/complex/ctanh.h>
#include <thrust/detail/complex/ctanhf.h>
#include <thrust/detail/complex/csqrt.h>
#include <thrust/detail/complex/csqrtf.h>
#include <thrust/detail/complex/catrig.h>
#include <thrust/detail/complex/catrigf.h>
#include <thrust/detail/complex/stream.h>
#include <thrust/detail/complex/arithmetic.inl>

#else
#include <complex>

namespace thrust
{
  using std::complex;
  using std::conj;
  using std::abs;
  using std::arg;
  using std::norm;
  using std::polar;
  using std::cos;
  using std::cosh;
  using std::exp;
  using std::log;
  using std::log10;
  using std::pow;
  using std::sin;
  using std::sinh;
  using std::sqrt;
  using std::tan;
  using std::tanh;

#if __cplusplus >= 201103L
  using std::acos;
  using std::asin;
  using std::atan;

  using std::acosh;
  using std::asinh;
  using std::atanh;
#else
  template <typename ValueType>
    inline complex<ValueType> acosh(const complex<ValueType>& z){
    thrust::complex<ValueType> ret((z.real() - z.imag()) * (z.real() + z.imag()) - ValueType(1.0),
				 ValueType(2.0) * z.real() * z.imag());    
    ret = sqrt(ret);
    if (z.real() < ValueType(0.0)){
      ret = -ret;
    }
    ret += z;
    ret = log(ret);
    if (ret.real() < ValueType(0.0)){
      ret = -ret;
    }
    return ret;
  }
  template <typename ValueType>
    inline complex<ValueType> asinh(const complex<ValueType>& z){
    return log(sqrt(z*z+ValueType(1))+z);
  }

  template <typename ValueType>
    inline complex<ValueType> atanh(const complex<ValueType>& z){
    ValueType imag2 = z.imag() *  z.imag();   
    ValueType n = ValueType(1.0) + z.real();
    n = imag2 + n * n;

    ValueType d = ValueType(1.0) - z.real();
    d = imag2 + d * d;
    complex<ValueType> ret(ValueType(0.25) * (::log(n) - ::log(d)),0);

    d = ValueType(1.0) -  z.real() * z.real() - imag2;

    ret.imag(ValueType(0.5) * ::atan2(ValueType(2.0) * z.imag(), d));
    return ret;
  }
#endif
  

  template <typename T>
    struct norm_type {
      typedef T type; 
    };
  
  template <typename T>
    struct norm_type< complex<T> > { 
    typedef T type;
  };


/*! \} // complex_numbers
 */

/*! \} // numerics
 */
}
#endif
