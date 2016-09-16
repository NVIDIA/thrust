#pragma once

#include <string>
#include <cstdlib>

#ifdef __GNUC__

// see http://gcc.gnu.org/onlinedocs/libstdc++/manual/ext_demangling.html
#include <cxxabi.h>

std::string demangle(const std::string &mangled)
{
  int status;
  char *realname = abi::__cxa_demangle(mangled.c_str(), 0, 0, &status);
  std::string result(realname);
  std::free(realname);

  return result;
}

#else
// MSVC doesn't mangle the result of typeid().name()
std::string demangle(const std::string &mangled)
{
  return mangled;
}
#endif

