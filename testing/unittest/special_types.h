#pragma once

#include <iostream>
#include <thrust/execution_policy.h>

template <typename T, unsigned int N>
struct FixedVector
{
    T data[N];
    
    THRUST_HOST_DEVICE
    FixedVector()
    {
        for(unsigned int i = 0; i < N; i++)
            data[i] = T();
    }

    THRUST_HOST_DEVICE
    FixedVector(T init)
    {
        for(unsigned int i = 0; i < N; i++)
            data[i] = init;
    }

    THRUST_HOST_DEVICE
    FixedVector operator+(const FixedVector& bs) const
    {
        FixedVector output;
        for(unsigned int i = 0; i < N; i++)
            output.data[i] = data[i] + bs.data[i];
        return output;
    }
    
    THRUST_HOST_DEVICE
    bool operator<(const FixedVector& bs) const
    {
        for(unsigned int i = 0; i < N; i++)
        {
            if(data[i] < bs.data[i])
                return true;
            else if(bs.data[i] < data[i])
                return false;
        }
        return false;
    }

    THRUST_HOST_DEVICE
    bool operator==(const FixedVector& bs) const
    {
        for(unsigned int i = 0; i < N; i++)
        {
            if(!(data[i] == bs.data[i]))
                return false;
        }
        return true;                
    }
};

template<typename Key, typename Value>
  struct key_value
{
  typedef Key   key_type;
  typedef Value value_type;

  THRUST_HOST_DEVICE
  key_value(void)
    : key(), value()
  {}

  THRUST_HOST_DEVICE
  key_value(key_type k, value_type v)
    : key(k), value(v)
  {}

  THRUST_HOST_DEVICE
  bool operator<(const key_value &rhs) const
  {
    return key < rhs.key;
  }

  THRUST_HOST_DEVICE
  bool operator>(const key_value &rhs) const
  {
    return key > rhs.key;
  }

  THRUST_HOST_DEVICE
  bool operator==(const key_value &rhs) const
  {
    return key == rhs.key && value == rhs.value;
  }

  THRUST_HOST_DEVICE
  bool operator!=(const key_value &rhs) const
  {
    return !operator==(rhs);
  }

  friend std::ostream &operator<<(std::ostream &os, const key_value &kv)
  {
    return os << "(" << kv.key << ", " << kv.value << ")";
  }

  key_type key;
  value_type value;
};

struct user_swappable
{
  inline THRUST_HOST_DEVICE
  user_swappable(bool swapped = false)
    : was_swapped(swapped)
  {}

  bool was_swapped;
};

inline THRUST_HOST_DEVICE
bool operator==(const user_swappable &x, const user_swappable &y)
{
  return x.was_swapped == y.was_swapped;
}

inline THRUST_HOST_DEVICE
void swap(user_swappable &x, user_swappable &y)
{
  x.was_swapped = true;
  y.was_swapped = false;
}

// Inheriting from classes in anonymous namespaces is not allowed.
// The anonymous namespace tests don't use these, so just disable them:
#ifndef THRUST_USE_ANON_NAMESPACE

class my_system : public THRUST_NS_QUALIFIER::device_execution_policy<my_system>
{
  public:
    my_system(int)
      : correctly_dispatched(false),
        num_copies(0)
    {}

    my_system(const my_system &other)
      : correctly_dispatched(false),
        num_copies(other.num_copies + 1)
    {}

    void validate_dispatch()
    {
      correctly_dispatched = (num_copies == 0);
    }

    bool is_valid()
    {
      return correctly_dispatched;
    }

  private:
    bool correctly_dispatched;

    // count the number of copies so that we can validate
    // that dispatch does not introduce any
    unsigned int num_copies;


    // disallow default construction
    my_system();
};

struct my_tag : THRUST_NS_QUALIFIER::device_execution_policy<my_tag> {};

#endif // THRUST_USE_ANON_NAMESPACE

namespace unittest
{


using THRUST_NS_QUALIFIER::detail::int8_t;
using THRUST_NS_QUALIFIER::detail::int16_t;
using THRUST_NS_QUALIFIER::detail::int32_t;
using THRUST_NS_QUALIFIER::detail::int64_t;

using THRUST_NS_QUALIFIER::detail::uint8_t;
using THRUST_NS_QUALIFIER::detail::uint16_t;
using THRUST_NS_QUALIFIER::detail::uint32_t;
using THRUST_NS_QUALIFIER::detail::uint64_t;

  
}

