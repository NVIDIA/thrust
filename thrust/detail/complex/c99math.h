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


#include <thrust/detail/complex/math_private.h>

namespace thrust
{
namespace detail
{
namespace complex
{
  /*
   * Define basic arithmetic functions so we can use them without explicit scope
   * keeping the code as close as possible to FreeBSDs for ease of maintenance. 
   * It also provides an easy way to support compilers with missing C99 functions.
   */
  using ::log;
  using ::acos;
  using ::asin;
  using ::sqrt;
  using ::sinh;
  using ::tan;
  using ::cos;
  using ::sin;
  using ::exp;
  using ::cosh;
  using ::atan;
  
#if __cplusplus >= 201103L
  using std::isinf;
  using std::isnan;
  using std::signbit;
  using std::isfinite;
  using std::atanh;
#elif defined _MSC_VER
  __host__ __device__ inline int isinf(float x){
    return std::abs(x) == 1.0f/0.0f;
  }

  __host__ __device__ inline int isinf(double x){
    return std::abs(x) == 1.0/0.0;
  }

  __host__ __device__ inline int isnan(float x){
    return x != x;
  }

  __host__ __device__ inline int isnan(double x){
    return x != x;
  }

  __host__ __device__ inline int signbit(float x){
    return (*((uint32_t *)&x)) & 0x80000000;
  }

  __host__ __device__ inline int signbit(double x){
    return (*((uint32_t *)&x)) & 0x80000000;
  }

  __host__ __device__ inline int isfinite(float x){
    return !isnan(x) && !isinf(x);
  }

  __host__ __device__ inline int isfinite(double x){
    return !isnan(x) && !isinf(x);
  }

  __host__ __device__ inline double atanh(double x){
    
  }

  __host__ __device__ inline double atanhf(double x){
    
  }
#else
  using ::atanh;
#endif
  
#if defined _MSC_VER
  __host__ __device__ inline int copysign(double x, double y){
    uint32_t hx,hy;
    get_high_word(hx,x);
    get_high_word(hy,y);
    set_high_word(x,(hx&0x7fffffff)|(hy&0x80000000));
    return x;
  }
  __host__ __device__ inline int copysignf(float x, float y){
    uint32_t ix,iy;
    get_float_word(ix,x);
    get_float_word(iy,y);
    set_float_word(x,(ix&0x7fffffff)|(iy&0x80000000));
    return x;
  }
}    
#endif

}
}
}
      
