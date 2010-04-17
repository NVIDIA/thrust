#pragma once

// for demangling the result of type_info.name()
// with msvc, type_info.name() is already demangled
#ifdef __GNUC__
#include <cxxabi.h>
#endif // __GNUC__

namespace unittest
{

#ifdef _CXXABI_H
inline const char* demangle(const char* name)
{
  char buf[1024];
  size_t size=1024;
  int status;
  char* res = abi::__cxa_demangle (name,
                               buf,
                               &size,
                               &status);
  return res;
}
#else
inline const char* demangle(const char* name)
{
  return name;
}
#endif

} // end unittest

