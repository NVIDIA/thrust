/*
 *  Copyright 2008-2010 NVIDIA Corporation
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

#include <thrust/random/uniform_real_distribution.h>
#include <math.h>


#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
// XXX WAR missing definitions on MSVC
__host__ __device__
double nextafter(double, double);

__host__ __device__
float nextafterf(float, float);
#endif // THRUST_HOST_COMPILER_MSVC


namespace thrust
{

namespace random
{


template<typename RealType>
  uniform_real_distribution<RealType>
    ::uniform_real_distribution(RealType a, RealType b)
      :m_param(a,b)
{
} // end uniform_real_distribution::uniform_real_distribution()

template<typename RealType>
  uniform_real_distribution<RealType>
    ::uniform_real_distribution(const param_type &parm)
      :m_param(parm)
{
} // end uniform_real_distribution::uniform_real_distribution()

template<typename RealType>
  void uniform_real_distribution<RealType>
    ::reset(void)
{
} // end uniform_real_distribution::reset()

template<typename RealType>
  template<typename UniformRandomNumberGenerator>
    typename uniform_real_distribution<RealType>::result_type
      uniform_real_distribution<RealType>
        ::operator()(UniformRandomNumberGenerator &urng)
{
  return operator()(urng, m_param);
} // end uniform_real::operator()()

namespace detail
{

__host__ __device__ __inline__
double nextafter(double x, double y)
{
  return ::nextafter(x,y);
}

__host__ __device__ __inline__
float nextafter(float x, float y)
{
  return ::nextafterf(x,y);
}

}

template<typename RealType>
  template<typename UniformRandomNumberGenerator>
    typename uniform_real_distribution<RealType>::result_type
      uniform_real_distribution<RealType>
        ::operator()(UniformRandomNumberGenerator &urng,
                     const param_type &parm)
{
  // call the urng & map its result to [0,1]
  result_type result = static_cast<result_type>(urng() - UniformRandomNumberGenerator::min);
  result /= static_cast<result_type>(UniformRandomNumberGenerator::max - UniformRandomNumberGenerator::min);

  // do not include parm.second in the result
  // get the next value after parm.second in the direction of parm.first
  // we need to do this because the range is half-open at parm.second

  return (result * (detail::nextafter(parm.second,parm.first) - parm.first)) + parm.first;
} // end uniform_real::operator()()

template<typename RealType>
  typename uniform_real_distribution<RealType>::result_type
    uniform_real_distribution<RealType>
      ::a(void) const
{
  return m_param.first;
} // end uniform_real::a()

template<typename RealType>
  typename uniform_real_distribution<RealType>::result_type
    uniform_real_distribution<RealType>
      ::b(void) const
{
  return m_param.second;
} // end uniform_real_distribution::b()

template<typename RealType>
  typename uniform_real_distribution<RealType>::param_type
    uniform_real_distribution<RealType>
      ::param(void) const
{
  return m_param;;
} // end uniform_real_distribution::param()

template<typename RealType>
  void uniform_real_distribution<RealType>
    ::param(const param_type &parm)
{
  m_param = parm;
} // end uniform_real_distribution::param()

template<typename RealType>
  typename uniform_real_distribution<RealType>::result_type
    uniform_real_distribution<RealType>
      ::min(void) const
{
  return a();
} // end uniform_real_distribution::min()

template<typename RealType>
  typename uniform_real_distribution<RealType>::result_type
    uniform_real_distribution<RealType>
      ::max(void) const
{
  return b();
} // end uniform_real_distribution::max()


template<typename RealType>
  bool uniform_real_distribution<RealType>
    ::equal(const uniform_real_distribution &rhs) const
{
  return m_param == rhs.param();
}


template<typename RealType>
  template<typename CharT, typename Traits>
    std::basic_ostream<CharT,Traits>&
      uniform_real_distribution<RealType>
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

  os << a() << space << b();

  // restore old flags and fill character
  os.flags(flags);
  os.fill(fill);
  return os;
}


template<typename RealType>
  template<typename CharT, typename Traits>
    std::basic_istream<CharT,Traits>&
      uniform_real_distribution<RealType>
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
bool operator==(const uniform_real_distribution<RealType> &lhs,
                const uniform_real_distribution<RealType> &rhs)
{
  return thrust::random::detail::random_core_access::equal(lhs,rhs);
}


template<typename RealType>
bool operator!=(const uniform_real_distribution<RealType> &lhs,
                const uniform_real_distribution<RealType> &rhs)
{
  return !(lhs == rhs);
}


template<typename RealType,
         typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os,
           const uniform_real_distribution<RealType> &d)
{
  return thrust::random::detail::random_core_access::stream_out(os,d);
}


template<typename RealType,
         typename CharT, typename Traits>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits> &is,
           uniform_real_distribution<RealType> &d)
{
  return thrust::random::detail::random_core_access::stream_in(is,d);
}


} // end random

} // end thrust

