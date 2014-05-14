/*
 *  Copyright 2008-2014 NVIDIA Corporation
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


/*! \file tuple_io.h
 *  \brief Provides streaming capabilities for thrust::tuple
 */

/*
 * Copyright (C) 2001 Jaakko Järvi (jaakko.jarvi@cs.utu.fi)
 *               2001 Gary Powell  (gary.powell@sierra.com)
 * 
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
 */

#pragma once

#include <istream>
#include <ostream>
#include <locale> // for two-arg isspace

#include <thrust/tuple.h>

namespace thrust {
namespace detail {
namespace tuple_detail {

class format_info {
public:   

   enum manipulator_type { open, close, delimiter };
   enum { number_of_manipulators = delimiter + 1 };
private:
   
   static int get_stream_index (int m)
   {
     static const int stream_index[number_of_manipulators]
        = { std::ios::xalloc(), std::ios::xalloc(), std::ios::xalloc() };

     return stream_index[m];
   }

   format_info(const format_info&);
   format_info();   


public:

   template<class CharType, class CharTrait>
   static CharType get_manipulator(std::basic_ios<CharType, CharTrait>& i, 
                                   manipulator_type m) {
     // The manipulators are stored as long.
     // A valid instanitation of basic_stream allows CharType to be any POD,
     // hence, the static_cast may fail (it fails if long is not convertible 
     // to CharType
     CharType c = static_cast<CharType>(i.iword(get_stream_index(m)) ); 
     // parentheses and space are the default manipulators
     if (!c) {
       switch(m) {
         case detail::tuple_detail::format_info::open :  c = i.widen('('); break;
         case detail::tuple_detail::format_info::close : c = i.widen(')'); break;
         case detail::tuple_detail::format_info::delimiter : c = i.widen(' '); break;
       }
     }
     return c;
   }


   template<class CharType, class CharTrait>
   static void set_manipulator(std::basic_ios<CharType, CharTrait>& i, 
                               manipulator_type m, CharType c) {
     // The manipulators are stored as long.
     // A valid instanitation of basic_stream allows CharType to be any POD,
     // hence, the static_cast may fail (it fails if CharType is not 
     // convertible long.
      i.iword(get_stream_index(m)) = static_cast<long>(c);
   }
};


template<class CharType>
class tuple_manipulator {
  const format_info::manipulator_type mt;
  CharType f_c;
public:
  explicit tuple_manipulator(format_info::manipulator_type m, 
                             const char c = 0)
     : mt(m), f_c(c) {}
  
  template<class CharTrait>
  void set(std::basic_ios<CharType, CharTrait> &io) const {
     format_info::set_manipulator(io, mt, f_c);
  }
};

template<class CharType, class CharTrait>
inline std::basic_ostream<CharType, CharTrait>&
operator<<(std::basic_ostream<CharType, CharTrait>& o, const tuple_manipulator<CharType>& m) {
  m.set(o);
  return o;
}


template<class CharType, class CharTrait>
inline std::basic_istream<CharType, CharTrait>&
operator>>(std::basic_istream<CharType, CharTrait>& i, const tuple_manipulator<CharType>& m) {
  m.set(i);
  return i;
}


} // end namespace tuple_detail
} // end namespace detail


template<class CharType>
inline detail::tuple_detail::tuple_manipulator<CharType> set_open(const CharType c) {
   return detail::tuple_detail::tuple_manipulator<CharType>(detail::tuple_detail::format_info::open, c);
}


template<class CharType>
inline detail::tuple_detail::tuple_manipulator<CharType> set_close(const CharType c) {
   return detail::tuple_detail::tuple_manipulator<CharType>(detail::tuple_detail::format_info::close, c);
}


template<class CharType>
inline detail::tuple_detail::tuple_manipulator<CharType> set_delimiter(const CharType c) {
   return detail::tuple_detail::tuple_manipulator<CharType>(detail::tuple_detail::format_info::delimiter, c);
}


namespace detail {
namespace tuple_detail {


// -------------------------------------------------------------
// printing tuples to ostream in format (a b c)
// parentheses and space are defaults, but can be overriden with manipulators
// set_open, set_close and set_delimiter


template<class CharType, class CharTrait>
inline std::basic_ostream<CharType, CharTrait>&
print(std::basic_ostream<CharType, CharTrait>& o, const thrust::tuple<>&) {
    return o;
}


template<class CharType, class CharTrait, class T0>
inline std::basic_ostream<CharType, CharTrait>&
print(std::basic_ostream<CharType, CharTrait>& o, const thrust::tuple<T0>& t) {

    const CharType d = detail::tuple_detail::format_info::get_manipulator(o, detail::tuple_detail::format_info::delimiter);

    o << thrust::get<0>(t);

    return o;
}


template<class CharType, class CharTrait, class T0, class T1>
inline std::basic_ostream<CharType, CharTrait>&
print(std::basic_ostream<CharType, CharTrait>& o, const thrust::tuple<T0, T1>& t) {

    const CharType d = detail::tuple_detail::format_info::get_manipulator(o, detail::tuple_detail::format_info::delimiter);

    o << thrust::get<0>(t) << d;
    o << thrust::get<1>(t);

    return o;
}


template<class CharType, class CharTrait, class T0, class T1, class T2>
inline std::basic_ostream<CharType, CharTrait>&
print(std::basic_ostream<CharType, CharTrait>& o, const thrust::tuple<T0, T1, T2>& t) {

    const CharType d = detail::tuple_detail::format_info::get_manipulator(o, detail::tuple_detail::format_info::delimiter);

    o << thrust::get<0>(t) << d;
    o << thrust::get<1>(t) << d;
    o << thrust::get<2>(t);

    return o;
}


template<class CharType, class CharTrait, class T0, class T1, class T2, class T3>
inline std::basic_ostream<CharType, CharTrait>&
print(std::basic_ostream<CharType, CharTrait>& o, const thrust::tuple<T0, T1, T2, T3>& t) {

    const CharType d = detail::tuple_detail::format_info::get_manipulator(o, detail::tuple_detail::format_info::delimiter);

    o << thrust::get<0>(t) << d;
    o << thrust::get<1>(t) << d;
    o << thrust::get<2>(t) << d;
    o << thrust::get<3>(t);

    return o;
}


template<class CharType, class CharTrait, class T0, class T1, class T2, class T3, class T4>
inline std::basic_ostream<CharType, CharTrait>&
print(std::basic_ostream<CharType, CharTrait>& o, const thrust::tuple<T0, T1, T2, T3, T4>& t) {

    const CharType d = detail::tuple_detail::format_info::get_manipulator(o, detail::tuple_detail::format_info::delimiter);

    o << thrust::get<0>(t) << d;
    o << thrust::get<1>(t) << d;
    o << thrust::get<2>(t) << d;
    o << thrust::get<3>(t) << d;
    o << thrust::get<4>(t);

    return o;
}


template<class CharType, class CharTrait, class T0, class T1, class T2, class T3, class T4, class T5>
inline std::basic_ostream<CharType, CharTrait>&
print(std::basic_ostream<CharType, CharTrait>& o, const thrust::tuple<T0, T1, T2, T3, T4, T5>& t) {

    const CharType d = detail::tuple_detail::format_info::get_manipulator(o, detail::tuple_detail::format_info::delimiter);

    o << thrust::get<0>(t) << d;
    o << thrust::get<1>(t) << d;
    o << thrust::get<2>(t) << d;
    o << thrust::get<3>(t) << d;
    o << thrust::get<4>(t) << d;
    o << thrust::get<5>(t);

    return o;
}


template<class CharType, class CharTrait, class T0, class T1, class T2, class T3, class T4, class T5, class T6>
inline std::basic_ostream<CharType, CharTrait>&
print(std::basic_ostream<CharType, CharTrait>& o, const thrust::tuple<T0, T1, T2, T3, T4, T5, T6>& t) {

    const CharType d = detail::tuple_detail::format_info::get_manipulator(o, detail::tuple_detail::format_info::delimiter);

    o << thrust::get<0>(t) << d;
    o << thrust::get<1>(t) << d;
    o << thrust::get<2>(t) << d;
    o << thrust::get<3>(t) << d;
    o << thrust::get<4>(t) << d;
    o << thrust::get<5>(t) << d;
    o << thrust::get<6>(t);

    return o;
}


template<class CharType, class CharTrait, class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
inline std::basic_ostream<CharType, CharTrait>&
print(std::basic_ostream<CharType, CharTrait>& o, const thrust::tuple<T0, T1, T2, T3, T4, T5, T6, T7>& t) {

    const CharType d = detail::tuple_detail::format_info::get_manipulator(o, detail::tuple_detail::format_info::delimiter);

    o << thrust::get<0>(t) << d;
    o << thrust::get<1>(t) << d;
    o << thrust::get<2>(t) << d;
    o << thrust::get<3>(t) << d;
    o << thrust::get<4>(t) << d;
    o << thrust::get<5>(t) << d;
    o << thrust::get<6>(t) << d;
    o << thrust::get<7>(t);

    return o;
}


template<class CharType, class CharTrait, class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8>
inline std::basic_ostream<CharType, CharTrait>&
print(std::basic_ostream<CharType, CharTrait>& o, const thrust::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8>& t) {

    const CharType d = detail::tuple_detail::format_info::get_manipulator(o, detail::tuple_detail::format_info::delimiter);

    o << thrust::get<0>(t) << d;
    o << thrust::get<1>(t) << d;
    o << thrust::get<2>(t) << d;
    o << thrust::get<3>(t) << d;
    o << thrust::get<4>(t) << d;
    o << thrust::get<5>(t) << d;
    o << thrust::get<6>(t) << d;
    o << thrust::get<7>(t) << d;
    o << thrust::get<8>(t);

    return o;
}


template<class CharType, class CharTrait, class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9>
inline std::basic_ostream<CharType, CharTrait>&
print(std::basic_ostream<CharType, CharTrait>& o, const thrust::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>& t) {

    const CharType d = detail::tuple_detail::format_info::get_manipulator(o, detail::tuple_detail::format_info::delimiter);

    o << thrust::get<0>(t) << d;
    o << thrust::get<1>(t) << d;
    o << thrust::get<2>(t) << d;
    o << thrust::get<3>(t) << d;
    o << thrust::get<4>(t) << d;
    o << thrust::get<5>(t) << d;
    o << thrust::get<6>(t) << d;
    o << thrust::get<7>(t) << d;
    o << thrust::get<8>(t) << d;
    o << thrust::get<9>(t);

    return o;
}


template<class CharT, class Traits, class T>
inline bool handle_width(std::basic_ostream<CharT, Traits>& o, const T& t) {
    std::streamsize width = o.width();
    if(width == 0) return false;

    std::basic_ostringstream<CharT, Traits> ss;

    ss.copyfmt(o);
    ss.tie(0);
    ss.width(0);

    ss << t;
    o << ss.str();

    return true;
}


} // end namespace tuple_detail
} // end namespace detail


template<class CharType, class CharTrait, class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9>
inline std::basic_ostream<CharType, CharTrait>& 
operator<<(std::basic_ostream<CharType, CharTrait>& o, 
           const thrust::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>& t) {
  if (!o.good() ) return o;
  if (detail::tuple_detail::handle_width(o, t)) return o;

  const CharType l = 
    detail::tuple_detail::format_info::get_manipulator(o, detail::tuple_detail::format_info::open);
  const CharType r = 
    detail::tuple_detail::format_info::get_manipulator(o, detail::tuple_detail::format_info::close);

  o << l;

  detail::tuple_detail::print(o, t);

  o << r;

  return o;
}


// -------------------------------------------------------------
// input stream operators

namespace detail {
namespace tuple_detail {

template<class CharType, class CharTrait>
inline std::basic_istream<CharType, CharTrait>& 
extract_and_check_delimiter(
  std::basic_istream<CharType, CharTrait> &is, format_info::manipulator_type del)
{
  const CharType d = format_info::get_manipulator(is, del);

  const bool is_delimiter = (!std::isspace(d, is.getloc()) );

  CharType c;
  if (is_delimiter) { 
    is >> c;
    if (is.good() && c!=d) { 
      is.setstate(std::ios::failbit);
    }
  } else {
    is >> std::ws;
  }
  return is;
}

   
template<class CharType, class CharTrait>
inline  std::basic_istream<CharType, CharTrait> & 
read (std::basic_istream<CharType, CharTrait> &is, thrust::tuple<>& t1) {

  return is;
}


template<class CharType, class CharTrait, class T0>
inline std::basic_istream<CharType, CharTrait>&
read(std::basic_istream<CharType, CharTrait> &is, thrust::tuple<T0>& t1) {

    if (!is.good()) return is;

    is >> thrust::get<0>(t1);

    return is;
}


template<class CharType, class CharTrait, class T0, class T1>
inline std::basic_istream<CharType, CharTrait>&
read(std::basic_istream<CharType, CharTrait> &is, thrust::tuple<T0, T1>& t1) {

    if (!is.good()) return is;

    is >> thrust::get<0>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<1>(t1);

    return is;
}


template<class CharType, class CharTrait, class T0, class T1, class T2>
inline std::basic_istream<CharType, CharTrait>&
read(std::basic_istream<CharType, CharTrait> &is, thrust::tuple<T0, T1, T2>& t1) {

    if (!is.good()) return is;

    is >> thrust::get<0>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<1>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<2>(t1);

    return is;
}


template<class CharType, class CharTrait, class T0, class T1, class T2, class T3>
inline std::basic_istream<CharType, CharTrait>&
read(std::basic_istream<CharType, CharTrait> &is, thrust::tuple<T0, T1, T2, T3>& t1) {

    if (!is.good()) return is;

    is >> thrust::get<0>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<1>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<2>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<3>(t1);

    return is;
}


template<class CharType, class CharTrait, class T0, class T1, class T2, class T3, class T4>
inline std::basic_istream<CharType, CharTrait>&
read(std::basic_istream<CharType, CharTrait> &is, thrust::tuple<T0, T1, T2, T3, T4>& t1) {

    if (!is.good()) return is;

    is >> thrust::get<0>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<1>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<2>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<3>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<4>(t1);

    return is;
}


template<class CharType, class CharTrait, class T0, class T1, class T2, class T3, class T4, class T5>
inline std::basic_istream<CharType, CharTrait>&
read(std::basic_istream<CharType, CharTrait> &is, thrust::tuple<T0, T1, T2, T3, T4, T5>& t1) {

    if (!is.good()) return is;

    is >> thrust::get<0>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<1>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<2>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<3>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<4>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<5>(t1);

    return is;
}


template<class CharType, class CharTrait, class T0, class T1, class T2, class T3, class T4, class T5, class T6>
inline std::basic_istream<CharType, CharTrait>&
read(std::basic_istream<CharType, CharTrait> &is, thrust::tuple<T0, T1, T2, T3, T4, T5, T6>& t1) {

    if (!is.good()) return is;

    is >> thrust::get<0>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<1>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<2>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<3>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<4>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<5>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<6>(t1);

    return is;
}


template<class CharType, class CharTrait, class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
inline std::basic_istream<CharType, CharTrait>&
read(std::basic_istream<CharType, CharTrait> &is, thrust::tuple<T0, T1, T2, T3, T4, T5, T6, T7>& t1) {

    if (!is.good()) return is;

    is >> thrust::get<0>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<1>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<2>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<3>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<4>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<5>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<6>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<7>(t1);

    return is;
}


template<class CharType, class CharTrait, class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8>
inline std::basic_istream<CharType, CharTrait>&
read(std::basic_istream<CharType, CharTrait> &is, thrust::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8>& t1) {

    if (!is.good()) return is;

    is >> thrust::get<0>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<1>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<2>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<3>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<4>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<5>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<6>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<7>(t1); extract_and_check_delimiter(is, format_info::delimiter);
    is >> thrust::get<8>(t1);

    return is;
}


template<class CharType, class CharTrait, class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9>
inline std::basic_istream<CharType, CharTrait>& 
read(std::basic_istream<CharType, CharTrait> &is, thrust::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>& t1) {

  if (!is.good()) return is;
   
  is >> thrust::get<0>(t1); extract_and_check_delimiter(is, format_info::delimiter);
  is >> thrust::get<1>(t1); extract_and_check_delimiter(is, format_info::delimiter);
  is >> thrust::get<2>(t1); extract_and_check_delimiter(is, format_info::delimiter);
  is >> thrust::get<3>(t1); extract_and_check_delimiter(is, format_info::delimiter);
  is >> thrust::get<4>(t1); extract_and_check_delimiter(is, format_info::delimiter);
  is >> thrust::get<5>(t1); extract_and_check_delimiter(is, format_info::delimiter);
  is >> thrust::get<6>(t1); extract_and_check_delimiter(is, format_info::delimiter);
  is >> thrust::get<7>(t1); extract_and_check_delimiter(is, format_info::delimiter);
  is >> thrust::get<8>(t1); extract_and_check_delimiter(is, format_info::delimiter);
  is >> thrust::get<9>(t1);

  return is;
}


} // end namespace tuple_detail
} // end namespace detail


template<class CharType, class CharTrait, class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9>
inline std::basic_istream<CharType, CharTrait>& 
operator>>(std::basic_istream<CharType, CharTrait>& is, thrust::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>& t1) {

  if (!is.good() ) return is;

  detail::tuple_detail::extract_and_check_delimiter(is, detail::tuple_detail::format_info::open);
  
  detail::tuple_detail::read(is, t1);
   
  detail::tuple_detail::extract_and_check_delimiter(is, detail::tuple_detail::format_info::close);

  return is;
}


} // end of namespace thrust

