#include <unittest/unittest.h>
#include <thrust/memory.h>

struct my_tag : thrust::device_system_tag {};

template<typename T1, typename T2>
bool is_same(const T1 &, const T2 &)
{
  return false;
}

template<typename T>
bool is_same(const T &, const T &)
{
  return true;
}

void TestSelectSystemDifferentTypes()
{
  // XXX different systems handle this differently, so it's impossible to test it right now
  KNOWN_FAILURE;
}
DECLARE_UNITTEST(TestSelectSystemDifferentTypes);


void TestSelectSystemSameTypes()
{
  using thrust::system::detail::generic::select_system;

  // select_system(device_system_tag, device_system_tag) should return device_system_tag
  bool is_device_system_tag = is_same(thrust::device_system_tag(), select_system(thrust::device_system_tag(), thrust::device_system_tag()));
  ASSERT_EQUAL(true, is_device_system_tag);

  // select_system(my_tag, my_tag) should return my_tag
  bool is_my_tag = is_same(my_tag(), select_system(my_tag(), my_tag()));
  ASSERT_EQUAL(true, is_my_tag);
}
DECLARE_UNITTEST(TestSelectSystemSameTypes);

