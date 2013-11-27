#include <thrust/complex.h>

namespace thrust
{
  template<typename ValueType,class charT, class traits>
    std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& os, const complex<ValueType>& z)
    {
      os << '(' << z.real() << ',' << z.imag() << ')';
      return os;
    };

  template<typename ValueType, typename charT, class traits>
    std::basic_istream<charT, traits>&
    operator>>(std::basic_istream<charT, traits>& is, complex<ValueType>& z)
    {
      ValueType re, im;

      charT ch;
      is >> ch;

      if(ch == '(')
      {
        is >> re >> ch;
        if (ch == ',')
        {
          is >> im >> ch;
          if (ch == ')')
          {
            z = complex<ValueType>(re, im);
          }
          else
          {
            is.setstate(std::ios_base::failbit);
          }
        }
        else if (ch == ')')
        {
          z = re;
        }
        else
        {
          is.setstate(std::ios_base::failbit);
        }
      }
      else
      {
        is.putback(ch);
        is >> re;
        z = re;
      }
      return is;
    }
}
