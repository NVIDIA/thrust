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

