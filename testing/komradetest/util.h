#pragma once

#include "system.h"

namespace komradetest
{

template<typename T>
  const char *type_name(void)
{
  return demangle(typeid(T).name());
} // end type_name()

} // end komradetest

