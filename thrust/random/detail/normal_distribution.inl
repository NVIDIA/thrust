/*
 *
 *  Copyright 2008-2011 NVIDIA Corporation
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

#include <thrust/random/normal_distribution.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/detail/cstdint.h>
#include <thrust/detail/integer_traits.h>

namespace thrust
{

namespace random
{

namespace experimental
{


template<typename RealType>
  normal_distribution<RealType>
    ::normal_distribution(RealType a, RealType b)
      :m_param(a,b)
{
} // end normal_distribution::normal_distribution()


template<typename RealType>
  normal_distribution<RealType>
    ::normal_distribution(const param_type &parm)
      :m_param(parm)
{
} // end normal_distribution::normal_distribution()


template<typename RealType>
  void normal_distribution<RealType>
    ::reset(void)
{
} // end normal_distribution::reset()


template<typename RealType>
  template<typename UniformRandomNumberGenerator>
    typename normal_distribution<RealType>::result_type
      normal_distribution<RealType>
        ::operator()(UniformRandomNumberGenerator &urng)
{
  return operator()(urng, m_param);
} // end normal_distribution::operator()()


template<typename RealType>
  template<typename UniformRandomNumberGenerator>
    typename normal_distribution<RealType>::result_type
      normal_distribution<RealType>
        ::operator()(UniformRandomNumberGenerator &urng,
                     const param_type &parm)
{
  typedef typename UniformRandomNumberGenerator::result_type uint_type;
  const uint_type urng_range = UniformRandomNumberGenerator::max - UniformRandomNumberGenerator::min;

  // Constants for conversion
  const result_type S1 = static_cast<result_type>(1) / urng_range;
  const result_type S2 = S1 / 2;

  result_type S3 = static_cast<result_type>(-1.4142135623730950488016887242097); // -sqrt(2)
  
  // Get the integer value
  uint_type u = urng() - UniformRandomNumberGenerator::min;

  // Ensure the conversion to float will give a value in the range [0,0.5)
  if(u > (urng_range / 2))
  {
    u = urng_range - u;
    S3 = -S3;
  }

  // Convert to floating point in [0,0.5)
  result_type p = u*S1 + S2;

  // Apply inverse error function
  return parm.first + parm.second * S3 * erfcinv(2 * p);
} // end normal_distribution::operator()()


template<typename RealType>
  typename normal_distribution<RealType>::param_type
    normal_distribution<RealType>
      ::param(void) const
{
  return m_param;
} // end normal_distribution::param()


template<typename RealType>
  void normal_distribution<RealType>
    ::param(const param_type &parm)
{
  m_param = parm;
} // end normal_distribution::param()


template<typename RealType>
  typename normal_distribution<RealType>::result_type
    normal_distribution<RealType>
      ::min THRUST_PREVENT_MACRO_SUBSTITUTION (void) const
{
  // XXX this solution is pretty terrible
  const thrust::detail::uint32_t inf_as_int = 0x7f800000u;
  const float inf = *reinterpret_cast<const float*>(&inf_as_int);
  return result_type(-inf);
} // end normal_distribution::min()


template<typename RealType>
  typename normal_distribution<RealType>::result_type
    normal_distribution<RealType>
      ::max THRUST_PREVENT_MACRO_SUBSTITUTION (void) const
{
  // XXX this solution is pretty terrible
  const thrust::detail::uint32_t inf_as_int = 0x7f800000u;
  const float inf = *reinterpret_cast<const float*>(&inf_as_int);
  return result_type(inf);
} // end normal_distribution::max()


template<typename RealType>
  typename normal_distribution<RealType>::result_type
    normal_distribution<RealType>
      ::mean(void) const
{
  return m_param.first;
} // end normal_distribution::mean()


template<typename RealType>
  typename normal_distribution<RealType>::result_type
    normal_distribution<RealType>
      ::stddev(void) const
{
  return m_param.second;
} // end normal_distribution::stddev()


template<typename RealType>
  bool normal_distribution<RealType>
    ::equal(const normal_distribution &rhs) const
{
  return m_param == rhs.param();
}


template<typename RealType>
  template<typename CharT, typename Traits>
    std::basic_ostream<CharT,Traits>&
      normal_distribution<RealType>
        ::stream_out(std::basic_ostream<CharT,Traits> &os) const
{
  typedef std::basic_ostream<CharT,Traits> ostream_type;
  typedef typename ostream_type::ios_base  ios_base;

  // save old flags and fill character
  const typename ios_base::fmtflags flags = os.flags();
  const CharT fill = os.fill();

  const CharT space = os.widen(' ');
  os.flags(ios_base::dec | ios_base::fixed | ios_base::left);
  os.fill(space);

  os << mean() << space << stddev();

  // restore old flags and fill character
  os.flags(flags);
  os.fill(fill);
  return os;
}


template<typename RealType>
  template<typename CharT, typename Traits>
    std::basic_istream<CharT,Traits>&
      normal_distribution<RealType>
        ::stream_in(std::basic_istream<CharT,Traits> &is)
{
  typedef std::basic_istream<CharT,Traits> istream_type;
  typedef typename istream_type::ios_base  ios_base;

  // save old flags
  const typename ios_base::fmtflags flags = is.flags();

  is.flags(ios_base::skipws);

  is >> m_param.first >> m_param.second;

  // restore old flags
  is.flags(flags);
  return is;
}


template<typename RealType>
bool operator==(const normal_distribution<RealType> &lhs,
                const normal_distribution<RealType> &rhs)
{
  return thrust::random::detail::random_core_access::equal(lhs,rhs);
}


template<typename RealType>
bool operator!=(const normal_distribution<RealType> &lhs,
                const normal_distribution<RealType> &rhs)
{
  return !(lhs == rhs);
}


template<typename RealType,
         typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os,
           const normal_distribution<RealType> &d)
{
  return thrust::random::detail::random_core_access::stream_out(os,d);
}


template<typename RealType,
         typename CharT, typename Traits>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits> &is,
           normal_distribution<RealType> &d)
{
  return thrust::random::detail::random_core_access::stream_in(is,d);
}


} // end experimental

} // end random

} // end thrust

