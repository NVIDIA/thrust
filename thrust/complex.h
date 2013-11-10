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


namespace thrust
{

  template <typename ValueType> struct complex;


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
  

  // General Functions
  template<typename ValueType> __host__ __device__ ValueType abs(const complex<ValueType>& z);
  template<typename ValueType> __host__ __device__ ValueType arg(const complex<ValueType>& z);
  template<typename ValueType> __host__ __device__ ValueType norm(const complex<ValueType>& z);
  template<typename ValueType> __host__ __device__ complex<ValueType> conj(const complex<ValueType>& z);
  template<typename ValueType> __host__ __device__ complex<ValueType> polar(const ValueType& m, const ValueType& theta = 0);

  // Binary Arithmetic operators:
  template <typename ValueType> __host__ __device__ inline complex<ValueType> operator*(const complex<ValueType>& lhs, const complex<ValueType>& rhs);
  template <typename ValueType> __host__ __device__ inline complex<ValueType> operator*(const complex<ValueType>& lhs, const ValueType & rhs);
  template <typename ValueType> __host__ __device__ inline complex<ValueType> operator*(const ValueType& lhs, const complex<ValueType>& rhs);

  template <typename ValueType> __host__ __device__ inline complex<ValueType> operator/(const complex<ValueType>& lhs, const complex<ValueType>& rhs);

  template <typename ValueType> __host__ __device__ inline complex<ValueType> operator+(const complex<ValueType>& lhs, const complex<ValueType>& rhs);
  template <typename ValueType> __host__ __device__ inline complex<ValueType> operator+(const complex<ValueType>& lhs, const ValueType & rhs);
  template <typename ValueType> __host__ __device__ inline complex<ValueType> operator+(const ValueType& lhs, const complex<ValueType>& rhs);

  template <typename ValueType> __host__ __device__ inline complex<ValueType> operator-(const complex<ValueType>& lhs, const complex<ValueType>& rhs);
  template <typename ValueType> __host__ __device__ inline complex<ValueType> operator-(const complex<ValueType>& lhs, const ValueType & rhs);
  template <typename ValueType> __host__ __device__ inline complex<ValueType> operator-(const ValueType& lhs, const complex<ValueType>& rhs);

  // Unary Arithmetic operators:
  template <typename ValueType> __host__ __device__ inline complex<ValueType> operator+(const complex<ValueType>& rhs);
  template <typename ValueType> __host__ __device__ inline complex<ValueType> operator-(const complex<ValueType>& rhs);

  // Exponential Functions:
  template <typename ValueType> __host__ __device__ complex<ValueType> exp(const complex<ValueType>& z);
  template <typename ValueType> __host__ __device__ complex<ValueType> log(const complex<ValueType>& z);
  template <typename ValueType> __host__ __device__ complex<ValueType> log10(const complex<ValueType>& z);

  // Power Functions:
  template <typename ValueType> __host__ __device__ complex<ValueType> pow(const complex<ValueType>& z, const complex<ValueType>& z2);
  template <typename ValueType> __host__ __device__ complex<ValueType> pow(const complex<ValueType>& z, const ValueType& x);
  template <typename ValueType> __host__ __device__ complex<ValueType> pow(const ValueType& x, const complex<ValueType>& z);
  template <typename T, typename U> __host__ __device__ complex<typename detail::promoted_numerical_type<T,U>::type > pow(const complex<T>& z, const U& x);
  template <typename T, typename U> __host__ __device__ complex<typename detail::promoted_numerical_type<T,U>::type > pow(const T& x,const complex<U>& z);
  template <typename T, typename U> __host__ __device__ complex<typename detail::promoted_numerical_type<T,U>::type > pow(const complex<T>& z, const complex<U>& z2);
  template <typename ValueType> __host__ __device__ complex<ValueType> sqrt(const complex<ValueType>&z);

  // Trigonometric functions:
  template <typename ValueType> __host__ __device__ complex<ValueType> cos(const complex<ValueType>& z);
  template <typename ValueType> __host__ __device__ complex<ValueType> sin(const complex<ValueType>&z);
  template <typename ValueType> __host__ __device__ complex<ValueType> tan(const complex<ValueType>&z);

  // Hyperbolic functions
  template <typename ValueType> __host__ __device__ complex<ValueType> cosh(const complex<ValueType>& z);
  template <typename ValueType> __host__ __device__ complex<ValueType> sinh(const complex<ValueType>&z);
  template <typename ValueType> __host__ __device__ complex<ValueType> tanh(const complex<ValueType>&z);

  // Inverse Trigonometric functions:
  template <typename ValueType> __host__ __device__ complex<ValueType> acos(const complex<ValueType>& z);
  template <typename ValueType> __host__ __device__ complex<ValueType> asin(const complex<ValueType>& z);
  template <typename ValueType> __host__ __device__ complex<ValueType> atan(const complex<ValueType>& z);

  // Inverse Hyperbolic functions:
  template <typename ValueType> __host__ __device__ complex<ValueType> acosh(const complex<ValueType>& z);
  template <typename ValueType> __host__ __device__ complex<ValueType> asinh(const complex<ValueType>& z);
  template <typename ValueType> __host__ __device__ complex<ValueType> atanh(const complex<ValueType>& z);



  // Stream operators:
  template<typename ValueType,class charT, class traits>
    std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& os, const complex<ValueType>& z);
  template<typename ValueType, typename charT, class traits>
    std::basic_istream<charT, traits>&
    operator>>(std::basic_istream<charT, traits>& is, complex<ValueType>& z);
  

  // Stream operators
  template<typename ValueType,class charT, class traits>
    std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& os, const complex<ValueType>& z)
    {
      os << '(' << z.real() << ',' << z.imag() << ')';
      return os;
    };

  template<typename ValueType, typename charT, class traits>
    std::basic_istream<charT, traits>&
    operator>>(std::basic_istream<charT, traits>& is, complex<ValueType>& z)
    {
      ValueType re, im;

      charT ch;
      is >> ch;

      if(ch == '(')
      {
        is >> re >> ch;
        if (ch == ',')
        {
          is >> im >> ch;
          if (ch == ')')
          {
            z = complex<ValueType>(re, im);
          }
          else
          {
            is.setstate(std::ios_base::failbit);
          }
        }
        else if (ch == ')')
        {
          z = re;
        }
        else
        {
          is.setstate(std::ios_base::failbit);
        }
      }
      else
      {
        is.putback(ch);
        is >> re;
        z = re;
      }
      return is;
    }


  
template <typename ValueType>
struct complex
{
public:
  typedef ValueType value_type;
  ValueType _m[2];
  // Constructors
  __host__ __device__      
    inline complex<ValueType>(const ValueType & re = ValueType(), const ValueType& im = ValueType())
    {
      real(re);
      imag(im);
    }  

  template <class X> 
    __host__ __device__
    inline complex<ValueType>(const complex<X> & z)
    {
      real(z.real());
      imag(z.imag());
    }  
  
  template <class X> 
    __host__
    inline complex<ValueType>(const std::complex<X> & z)
    {
      real(z.real());
      imag(z.imag());
    }  

  template <typename T>
    __host__ __device__
    inline complex<ValueType>& operator=(const complex<T> z)
    {
      real(z.real());
      imag(z.imag());
      return *this;
    }

  __host__ __device__
    inline complex<ValueType>& operator+=(const complex<ValueType> z)
    {
      real(real()+z.real());
      imag(imag()+z.imag());
      return *this;
    }

  __host__ __device__
    inline complex<ValueType>& operator-=(const complex<ValueType> z)
    {
      real(real()-z.real());
      imag(imag()-z.imag());
      return *this;
    }

  __host__ __device__
    inline complex<ValueType>& operator*=(const complex<ValueType> z)
    {
      *this = *this * z;
      return *this;
    }

  __host__ __device__
    inline complex<ValueType>& operator/=(const complex<ValueType> z)
    {
      *this = *this / z;
      return *this;
    }


  __host__ __device__ inline ValueType real() const volatile{ return _m[0]; }
  __host__ __device__ inline ValueType imag() const volatile{ return _m[1]; }
  __host__ __device__ inline ValueType real() const{ return _m[0]; }
  __host__ __device__ inline ValueType imag() const{ return _m[1]; }
  __host__ __device__ inline void real(ValueType re)volatile{ _m[0] = re; }
  __host__ __device__ inline void imag(ValueType im)volatile{ _m[1] = im; }
  __host__ __device__ inline void real(ValueType re){ _m[0] = re; }
  __host__ __device__ inline void imag(ValueType im){ _m[1] = im; }

  // cast operators
  inline operator std::complex<ValueType>() const { return std::complex<ValueType>(real(),imag()); }
};



  // Binary arithmetic operations
  // At the moment I'm implementing the basic functions, and the 
  // corresponding cuComplex calls are commented.

  template<typename ValueType>
    __host__ __device__ 
    inline complex<ValueType> operator+(const complex<ValueType>& lhs,
					const complex<ValueType>& rhs){
    return complex<ValueType>(lhs.real()+rhs.real(),lhs.imag()+rhs.imag());
  }

  template<typename ValueType>
    __host__ __device__ 
    inline complex<ValueType> operator+(const volatile complex<ValueType>& lhs,
					const volatile complex<ValueType>& rhs){
    return complex<ValueType>(lhs.real()+rhs.real(),lhs.imag()+rhs.imag());
  }

  template <typename ValueType> 
    __host__ __device__ 
    inline complex<ValueType> operator+(const complex<ValueType>& lhs, const ValueType & rhs){
    return complex<ValueType>(lhs.real()+rhs,lhs.imag());
  }
  template <typename ValueType> 
    __host__ __device__ 
    inline complex<ValueType> operator+(const ValueType& lhs, const complex<ValueType>& rhs){
    return complex<ValueType>(rhs.real()+lhs,rhs.imag());
  }

  template <typename ValueType> 
    __host__ __device__ 
    inline complex<ValueType> operator-(const complex<ValueType>& lhs, const complex<ValueType>& rhs){
    return complex<ValueType>(lhs.real()-rhs.real(),lhs.imag()-rhs.imag());
  }
  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator-(const complex<ValueType>& lhs, const ValueType & rhs){
    return complex<ValueType>(lhs.real()-rhs,lhs.imag());
  }
  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator-(const ValueType& lhs, const complex<ValueType>& rhs){
    return complex<ValueType>(lhs-rhs.real(),-rhs.imag());
  }

  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator*(const complex<ValueType>& lhs,
					    const complex<ValueType>& rhs){
    return complex<ValueType>(lhs.real()*rhs.real()-lhs.imag()*rhs.imag(),
				  lhs.real()*rhs.imag()+lhs.imag()*rhs.real());
  }

  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator*(const complex<ValueType>& lhs, const ValueType & rhs){
    return complex<ValueType>(lhs.real()*rhs,lhs.imag()*rhs);
  }

  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator*(const ValueType& lhs, const complex<ValueType>& rhs){
    return complex<ValueType>(rhs.real()*lhs,rhs.imag()*lhs);
  }


  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator/(const complex<ValueType>& lhs, const complex<ValueType>& rhs){
    ValueType s = std::abs(rhs.real()) + std::abs(rhs.imag());
    ValueType oos = ValueType(1.0) / s;
    ValueType ars = lhs.real() * oos;
    ValueType ais = lhs.imag() * oos;
    ValueType brs = rhs.real() * oos;
    ValueType bis = rhs.imag() * oos;
    s = (brs * brs) + (bis * bis);
    oos = ValueType(1.0) / s;
    complex<ValueType> quot(((ars * brs) + (ais * bis)) * oos,
			 ((ais * brs) - (ars * bis)) * oos);
    return quot;
  }

  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator/(const complex<ValueType>& lhs, const ValueType & rhs){
    return complex<ValueType>(lhs.real()/rhs,lhs.imag()/rhs);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator/(const ValueType& lhs, const complex<ValueType>& rhs){
    return complex<ValueType>(lhs)/rhs;
  }


  // Unary arithmetic operations
  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator+(const complex<ValueType>& rhs){
    return rhs;
  }
  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator-(const complex<ValueType>& rhs){
    return rhs*-ValueType(1);
  }

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


  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> conj(const complex<ValueType>& z){
    return complex<ValueType>(z.real(),-z.imag());
  }


  // As std::hypot is only C++11 we have to use the C interface
  template <typename ValueType>
    __host__ __device__
    inline ValueType abs(const complex<ValueType>& z){
    return ::hypot(z.real(),z.imag());
  }
  template <>
    __host__ __device__
    inline float abs(const complex<float>& z){
    return ::hypotf(z.real(),z.imag());
  }
  template<>
    __host__ __device__
    inline double abs(const complex<double>& z){
    return ::hypot(z.real(),z.imag());
  }


  template <typename ValueType>
    __host__ __device__
    inline ValueType arg(const complex<ValueType>& z){
    return std::atan2(z.imag(),z.real());
  }

  template <typename ValueType>
    __host__ __device__
    inline ValueType norm(const complex<ValueType>& z){
    // not fast, but accurate
    return abs(z)*abs(z);
    //    return z.real()*z.real() + z.imag()*z.imag();
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> polar(const ValueType & m, const ValueType & theta){ 
    return complex<ValueType>(m * std::cos(theta),m * std::sin(theta));
  }

  // Transcendental functions implementation
  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> cos(const complex<ValueType>& z){
    const ValueType re = z.real();
    const ValueType im = z.imag();
    return complex<ValueType>(std::cos(re) * std::cosh(im), 
			      -std::sin(re) * std::sinh(im));
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> cosh(const complex<ValueType>& z){
    const ValueType re = z.real();
    const ValueType im = z.imag();
    return complex<ValueType>(std::cosh(re) * std::cos(im), 
			      std::sinh(re) * std::sin(im));
  }

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

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> sin(const complex<ValueType>& z){
    const ValueType re = z.real();
    const ValueType im = z.imag();
    return complex<ValueType>(std::sin(re) * std::cosh(im), 
			      std::cos(re) * std::sinh(im));
  }


  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> sinh(const complex<ValueType>& z){
    const ValueType re = z.real();
    const ValueType im = z.imag();
    return complex<ValueType>(std::sinh(re) * std::cos(im), 
			      std::cosh(re) * std::sin(im));
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> sqrt(const complex<ValueType>& z){
    return thrust::polar(std::sqrt(thrust::abs(z)),thrust::arg(z)/ValueType(2));
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> tan(const complex<ValueType>& z){
    return sin(z)/cos(z);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> tanh(const complex<ValueType>& z){
    // This implementation seems better than the simple sin/cos
    return (thrust::exp(ValueType(2)*z)-ValueType(1))/
      (thrust::exp(ValueType(2)*z)+ValueType(1));
  }

  // Inverse trigonometric functions implementation

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> acos(const complex<ValueType>& z){
    const complex<ValueType> ret = thrust::asin(z);
    return complex<ValueType>(ValueType(M_PI/2.0) - ret.real(),-ret.imag());
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> asin(const complex<ValueType>& z){
    const complex<ValueType> i(0,1);
    return -i*thrust::asinh(i*z);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> atan(const complex<ValueType>& z){
    const complex<ValueType> i(0,1);
    return -i*thrust::atanh(i*z);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> acosh(const complex<ValueType>& z){
    thrust::complex<ValueType> ret((z.real() - z.imag()) * (z.real() + z.imag()) - ValueType(1.0),
				   ValueType(2.0) * z.real() * z.imag());    
    ret = thrust::sqrt(ret);
    if (z.real() < ValueType(0.0)){
      ret = -ret;
    }
    ret += z;
    ret = thrust::log(ret);
    if (ret.real() < ValueType(0.0)){
      ret = -ret;
    }
    return ret;
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> asinh(const complex<ValueType>& z){
    return thrust::log(thrust::sqrt(z*z+ValueType(1))+z);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> atanh(const complex<ValueType>& z){
    ValueType imag2 = z.imag() *  z.imag();   
    ValueType n = ValueType(1.0) + z.real();
    n = imag2 + n * n;

    ValueType d = ValueType(1.0) - z.real();
    d = imag2 + d * d;
    complex<ValueType> ret(ValueType(0.25) * (std::log(n) - std::log(d)),0);

    d = ValueType(1.0) -  z.real() * z.real() - imag2;

    ret.imag(ValueType(0.5) * std::atan2(ValueType(2.0) * z.imag(), d));
    return ret;
  }

} // end namespace thrust

#include <thrust/detail/complex/cexp.h>
#include <thrust/detail/complex/cexpf.h>
#include <thrust/detail/complex/clog.h>
#include <thrust/detail/complex/clogf.h>

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
}
#endif
