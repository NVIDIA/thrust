#pragma once

#include "system.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace unittest
{

template<typename T>
  const char *type_name(void)
{
  return demangle(typeid(T).name());
} // end type_name()

} // end unittest

