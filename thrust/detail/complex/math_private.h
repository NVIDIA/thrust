/* adapted from FreeBSD:
 *    lib/msun/src/math_private.h
 *
 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

#pragma once

#include <thrust/complex.h>

namespace thrust
{
  namespace detail{
    namespace complex{

      using thrust::complex;

      typedef union
      {
	float value;
	uint32_t word;
      } ieee_float_shape_type;

      __host__ __device__
      inline void __get_float_word(uint32_t & i, float d){
	ieee_float_shape_type gf_u;
	gf_u.value = (d);
	(i) = gf_u.word;
      }

      __host__ __device__
      inline void __set_float_word(float & d, uint32_t i){
	ieee_float_shape_type sf_u;
	sf_u.word = (i);
	(d) = sf_u.value;
      }

      // Assumes little endian ordering
      typedef union
      {
	double value;
	struct
	{
	  uint32_t lsw;
	  uint32_t msw;
	} parts;
	struct
	{
	  uint64_t w;
	} xparts;
      } ieee_double_shape_type;
      
      __host__ __device__
      void __get_high_word(uint32_t & i,double d){
	ieee_double_shape_type gh_u;
	gh_u.value = (d);
	(i) = gh_u.parts.msw;                                   
      };
      
      __host__ __device__
      void  __insert_words(double & d, uint32_t ix0, uint32_t ix1){
	ieee_double_shape_type iw_u;
	iw_u.parts.msw = (ix0);
	iw_u.parts.lsw = (ix1);
	(d) = iw_u.value;
      }

    }
  }
}
