#pragma once

#include <thrust/complex.h>

namespace thrust
{

  /* --- Binary Arithmetic Operators --- */

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



  /* --- Unary  Arithmetic Operators --- */

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
}
