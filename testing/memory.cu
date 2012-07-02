#include <iostream>
#include <unittest/unittest.h>
#include <thrust/sort.h>
#include <thrust/memory.h>
#include <thrust/pair.h>

struct my_tag : thrust::device_system_state<my_tag> {};

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
  using thrust::system::detail::generic::select_system;

  // select_system(my_tag, device_system_tag) should return device_system_tag (the minimum tag)
  bool is_device_system_tag = is_same(thrust::device_system_tag(), select_system(my_tag(), thrust::device_system_tag()));
  ASSERT_EQUAL(true, is_device_system_tag);

  // select_system(device_system_tag, my_tag) should return device_system_tag (the minimum tag)
  is_device_system_tag = is_same(thrust::device_system_tag(), select_system(thrust::device_system_tag(), my_tag()));
  ASSERT_EQUAL(true, is_device_system_tag);
}
DECLARE_UNITTEST(TestSelectSystemDifferentTypes);


void TestSelectSystemSameTypes()
{
  using thrust::system::detail::generic::select_system;

  // select_system(host_system_tag, host_system_tag) should return host_system_tag
  bool is_host_system_tag = is_same(thrust::host_system_tag(), select_system(thrust::host_system_tag(), thrust::host_system_tag()));
  ASSERT_EQUAL(true, is_host_system_tag);

  // select_system(device_system_tag, device_system_tag) should return device_system_tag
  bool is_device_system_tag = is_same(thrust::device_system_tag(), select_system(thrust::device_system_tag(), thrust::device_system_tag()));
  ASSERT_EQUAL(true, is_device_system_tag);

  // select_system(my_tag, my_tag) should return my_tag
  bool is_my_tag = is_same(my_tag(), select_system(my_tag(), my_tag()));
  ASSERT_EQUAL(true, is_my_tag);
}
DECLARE_UNITTEST(TestSelectSystemSameTypes);


//template<typename T>
//  thrust::pair<thrust::pointer<T,my_tag>, std::ptrdiff_t>
//    get_temporary_buffer(my_tag, std::ptrdiff_t n)
//{
//  // communicate that my version of get_temporary_buffer
//  // was correctly dispatched
//  throw my_tag();
//}
//
void TestGetTemporaryBufferDispatchImplicit()
{
  KNOWN_FAILURE;

//  bool correctly_dispatched = false;
//
//  try
//  {
//    thrust::device_vector<int> vec(2);
//
//    // call something we know will invoke get_temporary_buffer
//    thrust::sort(thrust::retag<my_tag>(vec.begin()),
//                 thrust::retag<my_tag>(vec.end()));
//  }
//  catch(my_tag)
//  {
//    correctly_dispatched = true;
//  }
//
//  ASSERT_EQUAL(true, correctly_dispatched);
}
DECLARE_UNITTEST(TestGetTemporaryBufferDispatchImplicit);

