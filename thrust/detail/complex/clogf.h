/* adapted from FreeBSDs msun:*/
/*-
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */


#pragma once

#include <thrust/complex.h>
#include <thrust/detail/complex/math_private.h>

namespace thrust
{
  namespace detail{
    namespace complex{

      using thrust::complex;

      /* round down to 8 = 24/3 bits */
      __host__ __device__
	float __trim(float x){
	uint32_t hx;
	__get_float_word(hx, x);
	hx &= 0xffff0000;
	float ret;
	__set_float_word(ret,hx);
	return ret;
      }
  

      __host__ __device__
	complex<float> clogf(const complex<float>& z){

	// Adapted from FreeBSDs msun
	float x, y;
	float ax, ay;
	float x0, y0, x1, y1, x2, y2, t, hm1;
	float val[12];
	int i, sorted;
    
	x = z.real();
	y = z.imag();
    
	/* Handle NaNs using the general formula to mix them right. */
	if (x != x || y != y){
	  return (complex<float>(std::log(norm(z)), std::atan2(y, x)));
	}
    
	ax = std::abs(x);
	ay = std::abs(y);
	if (ax < ay) {
	  t = ax;
	  ax = ay;
	  ay = t;
	}

	/*
	 * To avoid unnecessary overflow, if x and y are very large, divide x
	 * and y by M_E, and then add 1 to the logarithm.  This depends on
	 * M_E being larger than sqrt(2).
	 * There is a potential loss of accuracy caused by dividing by M_E,
	 * but this case should happen extremely rarely.
	 */
	// For high values of ay -> hypotf(FLT_MAX,ay) = inf
	// We expect that for values at or below ay = 1e34f this should not happen
	if (ay > 1e34f){ 
	  return (complex<float>(std::log(hypotf(x / M_E, y / M_E)) + 1.0f, std::atan2(y, x)));
	}
	if (ax == 1.f) {
	  if (ay < 1e-19f){
	    return (complex<float>((ay * 0.5f) * ay, std::atan2(y, x)));
	  }
	  return (complex<float>(log1pf(ay * ay) * 0.5f, std::atan2(y, x)));
	}

	/*
	 * Because atan2 and hypot conform to C99, this also covers all the
	 * edge cases when x or y are 0 or infinite.
	 */
	if (ax < 1e-6f || ay < 1e-6f || ax > 1e6f || ay > 1e6f){
	  return (complex<float>(std::log(hypotf(x, y)), std::atan2(y, x)));
	}
    
	/* 
	 * From this point on, we don't need to worry about underflow or
	 * overflow in calculating ax*ax or ay*ay.
	 */
    
	/* Some easy cases. */
    
	if (ax >= 1.0f){
	  return (complex<float>(log1pf((ax-1.f)*(ax+1.f) + ay*ay) * 0.5f, atan2(y, x)));
	}

	if (ax*ax + ay*ay <= 0.7f){
	  return (complex<float>(std::log(ax*ax + ay*ay) * 0.5f, std::atan2(y, x)));
	}

	/*
	 * Take extra care so that ULP of real part is small if hypot(x,y) is
	 * moderately close to 1.
	 */


	x0 = __trim(ax);
	ax = ax-x0;
	x1 = __trim(ax);
	x2 = ax-x1;
	y0 = __trim(ay);
	ay = ay-y0;
	y1 = __trim(ay);
	y2 = ay-y1;
    
	val[0] = x0*x0;
	val[1] = y0*y0;
	val[2] = 2*x0*x1;
	val[3] = 2*y0*y1;
	val[4] = x1*x1;
	val[5] = y1*y1;
	val[6] = 2*x0*x2;
	val[7] = 2*y0*y2;
	val[8] = 2*x1*x2;
	val[9] = 2*y1*y2;
	val[10] = x2*x2;
	val[11] = y2*y2;
    
	/* Bubble sort. */

	do {
	  sorted = 1;
	  for (i=0;i<11;i++) {
	    if (val[i] < val[i+1]) {
	      sorted = 0;
	      t = val[i];
	      val[i] = val[i+1];
	      val[i+1] = t;
	    }
	  }
	} while (!sorted);
    
	hm1 = -1;
	for (i=0;i<12;i++){
	  hm1 += val[i];
	}
	return (complex<float>(0.5f * log1pf(hm1), atan2(y, x)));
      }

    }
  }
}
    