/*
 *  Copyright 2013 Filipe RNC Maia
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

#include <thrust/complex.h>

namespace thrust
{

  /* --- Constructors --- */

  template <typename T>
  inline __host__ __device__  complex<T>
  ::complex(const T & re, const T& im)
  {
    real(re);
    imag(im);
  } 

  template <typename T>
  template <typename X> 
  inline __host__ __device__ complex<T>
  ::complex(const complex<X> & z)
  {
    real(z.real());
    imag(z.imag());
  }  

  template <typename T>
  template <typename X> 
  inline __host__ complex<T>
  ::complex(const std::complex<X> & z)
  {
    real(z.real());
    imag(z.imag());
  }  



  /* --- Compound Assignment Operators --- */

  template <typename T>
  __host__ __device__  inline 
  complex<T>& complex<T>::operator+=(const complex<T> z)
  {
    real(real()+z.real());
    imag(imag()+z.imag());
    return *this;
  }

  template <typename T>
  __host__ __device__
  inline complex<T>& complex<T>::operator-=(const complex<T> z)
  {
    real(real()-z.real());
    imag(imag()-z.imag());
    return *this;
  }

  template <typename T>
  __host__ __device__
  inline complex<T>& complex<T>::operator*=(const complex<T> z)
  {
    *this = *this * z;
    return *this;
  }

  template <typename T>
  __host__ __device__
  inline complex<T>& complex<T>::operator/=(const complex<T> z)
  {
    *this = *this / z;
    return *this;
  }



  /* --- Equality Operators --- */

  template <typename T> 
    __host__ __device__
    inline bool operator==(const complex<T>& lhs, const complex<T>& rhs){
    if(lhs.real() == rhs.real() && lhs.imag() == rhs.imag()){
      return true;
    }
    return false;
  }

  template <typename T> 
    __host__ __device__
    inline bool operator==(const T & lhs, const complex<T>& rhs){
    if(lhs == rhs.real() && rhs.imag() == 0){
      return true;
    }
    return false;
  }

  template <typename T> 
    __host__ __device__
    inline bool operator==(const complex<T> & lhs, const T& rhs){
    if(lhs.real() == rhs && lhs.imag() == 0){
      return true;
    }
    return false;
  }

  template <typename T> 
    __host__ __device__
    inline bool operator!=(const complex<T>& lhs, const complex<T>& rhs){
    return !(lhs == rhs);
  }

  template <typename T> 
    __host__ __device__
    inline bool operator!=(const T & lhs, const complex<T>& rhs){
    return !(lhs == rhs);
  }

  template <typename T> 
    __host__ __device__
    inline bool operator!=(const complex<T> & lhs, const T& rhs){
    return !(lhs == rhs);
  }

} 

#include <thrust/detail/complex/arithmetic.inl>
#include <thrust/detail/complex/cexp.inl>
#include <thrust/detail/complex/cexpf.inl>
#include <thrust/detail/complex/clog.inl>
#include <thrust/detail/complex/clogf.inl>
#include <thrust/detail/complex/ccosh.inl>
#include <thrust/detail/complex/ccoshf.inl>
#include <thrust/detail/complex/csinh.inl>
#include <thrust/detail/complex/csinhf.inl>
#include <thrust/detail/complex/ctanh.inl>
#include <thrust/detail/complex/ctanhf.inl>
#include <thrust/detail/complex/csqrt.inl>
#include <thrust/detail/complex/csqrtf.inl>
#include <thrust/detail/complex/catrig.inl>
#include <thrust/detail/complex/catrigf.inl>
#include <thrust/detail/complex/stream.inl>

