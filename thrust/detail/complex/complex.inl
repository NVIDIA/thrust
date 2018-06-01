/*
 *  Copyright 2008-2013 NVIDIA Corporation
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
__host__ __device__
complex<T>::complex()
#if __cplusplus >= 201103L
  // Initialize the storage in the member initializer list using C++ unicorn
  // initialization. This allows `complex<T const>` to work.
  // We do a functional-style cast here to suppress conversion warnings.
  : data{T(), T()}
{}
#else
{
  real(T());
  imag(T());
} 
#endif

template <typename T>
__host__ __device__
complex<T>::complex(const T& re)
#if __cplusplus >= 201103L
  // Initialize the storage in the member initializer list using C++ unicorn
  // initialization. This allows `complex<T const>` to work.
  : data{re, T()}
{}
#else
{
  real(re);
  imag(T());
} 
#endif

#if 0
template <typename T>
template <typename R>
__host__ __device__
complex<T>::complex(const R& re)
#if __cplusplus >= 201103L
  // Initialize the storage in the member initializer list using C++ unicorn
  // initialization. This allows `complex<T const>` to work.
  // We do a functional-style cast here to suppress conversion warnings.
  : data{T(re), T()}
{}
#else
{
  real(T(re));
  imag(T());
} 
#endif
#endif

template <typename T>
__host__ __device__
complex<T>::complex(const T& re, const T& im)
#if __cplusplus >= 201103L
  // Initialize the storage in the member initializer list using C++ unicorn
  // initialization. This allows `complex<T const>` to work.
  : data{re, im}
{}
#else
{
  real(re);
  imag(im);
}
#endif 

#if 0
template <typename T>
template <typename R, typename I>
__host__ __device__
complex<T>::complex(const R& re, const I& im)
#if __cplusplus >= 201103L
  // Initialize the storage in the member initializer list using C++ unicorn
  // initialization. This allows `complex<T const>` to work.
  // We do a functional-style cast here to suppress conversion warnings.
  : data{T(re), T(im)}
{}
#else
{
  real(T(re));
  imag(T(im));
}
#endif 
#endif

template <typename T>
__host__ __device__
complex<T>::complex(const complex<T>& z)
#if __cplusplus >= 201103L
  // Initialize the storage in the member initializer list using C++ unicorn
  // initialization. This allows `complex<T const>` to work.
  : data{z.real(), z.imag()}
{}
#else
{
  real(z.real());
  imag(z.imag());
}
#endif 

template <typename T>
template <typename U> 
__host__ __device__
complex<T>::complex(const complex<U>& z)
#if __cplusplus >= 201103L
  // Initialize the storage in the member initializer list using C++ unicorn
  // initialization. This allows `complex<T const>` to work.
  // We do a functional-style cast here to suppress conversion warnings.
  : data{T(z.real()), T(z.imag())}
{}
#else
{
  real(T(z.real()));
  imag(T(z.imag()));
}
#endif 

template <typename T>
__host__
complex<T>::complex(const std::complex<T>& z)
#if __cplusplus >= 201103L
  // Initialize the storage in the member initializer list using C++ unicorn
  // initialization. This allows `complex<T const>` to work.
  : data{z.real(), z.imag()}
{}
#else
{
  real(z.real());
  imag(z.imag());
}  
#endif

template <typename T>
template <typename U> 
__host__
complex<T>::complex(const std::complex<U>& z)
#if __cplusplus >= 201103L
  // Initialize the storage in the member initializer list using C++ unicorn
  // initialization. This allows `complex<T const>` to work.
  // We do a functional-style cast here to suppress conversion warnings.
  : data{T(z.real()), T(z.imag())}
{}
#else
{
  real(T(z.real()));
  imag(T(z.imag()));
}  
#endif



/* --- Assignment Operators --- */

template <typename T>
__host__ __device__
complex<T>& complex<T>::operator=(const T& re)
{
  real(re);
  imag(T());
  return *this;
}

#if 0
template <typename T>
template <typename R>
__host__ __device__
complex<T>& complex<T>::operator=(const R& re)
{
  real(re);
  imag(T());
  return *this;
}
#endif

template <typename T>
complex<T>& complex<T>::operator=(const complex<T>& z)
{
  real(z.real());
  imag(z.imag());
  return *this;
}

template <typename T>
template <typename U>
__host__ __device__
complex<T>& complex<T>::operator=(const complex<U>& z)
{
  real(T(z.real()));
  imag(T(z.imag()));
  return *this;
}

template <typename T>
__host__
complex<T>& complex<T>::operator=(const std::complex<T>& z)
{
  real(z.real());
  imag(z.imag());
  return *this;
}

template <typename T>
template <typename U> 
__host__
complex<T>& complex<T>::operator=(const std::complex<U>& z)
{
  real(T(z.real()));
  imag(T(z.imag()));
  return *this;
}



/* --- Compound Assignment Operators --- */

template <typename T>
template <typename U> 
__host__ __device__ 
complex<T>& complex<T>::operator+=(const complex<U>& z)
{
  *this = *this + z;
  return *this;
}

template <typename T>
template <typename U> 
__host__ __device__
complex<T>& complex<T>::operator-=(const complex<U>& z)
{
  *this = *this - z;
  return *this;
}

template <typename T>
template <typename U> 
__host__ __device__
complex<T>& complex<T>::operator*=(const complex<U>& z)
{
  *this = *this * z;
  return *this;
}

template <typename T>
template <typename U> 
__host__ __device__
complex<T>& complex<T>::operator/=(const complex<U>& z)
{
  *this = *this / z;
  return *this;
}

template <typename T>
template <typename U> 
__host__ __device__ 
complex<T>& complex<T>::operator+=(const U& z)
{
  *this = *this + z;
  return *this;
}

template <typename T>
template <typename U> 
__host__ __device__
complex<T>& complex<T>::operator-=(const U& z)
{
  *this = *this - z;
  return *this;
}

template <typename T>
template <typename U> 
__host__ __device__
complex<T>& complex<T>::operator*=(const U& z)
{
  *this = *this * z;
  return *this;
}

template <typename T>
template <typename U> 
__host__ __device__
complex<T>& complex<T>::operator/=(const U& z)
{
  *this = *this / z;
  return *this;
}



/* --- Equality Operators --- */

template <typename T0, typename T1> 
__host__ __device__
bool operator==(const complex<T0>& x, const complex<T1>& y)
{
  return x.real() == y.real() && x.imag() == y.imag();
}

template <typename T0, typename T1> 
__host__ 
bool operator==(const complex<T0>& x, const std::complex<T1>& y)
{
  return x.real() == y.real() && x.imag() == y.imag();
}

template <typename T0, typename T1> 
__host__ 
bool operator==(const std::complex<T0>& x, const complex<T1>& y)
{
  return x.real() == y.real() && x.imag() == y.imag();
}

template <typename T0, typename T1> 
__host__ __device__
bool operator==(const T0& x, const complex<T1>& y)
{
  return x == y.real() && y.imag() == T1();
}

template <typename T0, typename T1> 
__host__ __device__
bool operator==(const complex<T0>& x, const T1& y)
{
  return x.real() == y && x.imag() == T1();
}

template <typename T0, typename T1> 
__host__ __device__
bool operator!=(const complex<T0>& x, const complex<T1>& y)
{
  return !(x == y);
}

template <typename T0, typename T1> 
__host__
bool operator!=(const complex<T0>& x, const std::complex<T1>& y)
{
  return !(x == y);
}

template <typename T0, typename T1> 
__host__
bool operator!=(const std::complex<T0>& x, const complex<T1>& y)
{
  return !(x == y);
}

template <typename T0, typename T1> 
__host__ __device__
bool operator!=(const T0& x, const complex<T1>& y)
{
  return !(x == y);
}

template <typename T0, typename T1> 
__host__ __device__
bool operator!=(const complex<T0>& x, const T1& y)
{
  return !(x == y);
}

} // end namespace thrust

#include <thrust/detail/complex/arithmetic.h>
#include <thrust/detail/complex/cproj.h>
#include <thrust/detail/complex/cexp.h>
#include <thrust/detail/complex/cexpf.h>
#include <thrust/detail/complex/clog.h>
#include <thrust/detail/complex/clogf.h>
#include <thrust/detail/complex/cpow.h>
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

