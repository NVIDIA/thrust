#include <unittest/unittest.h>
#include <thrust/sort.h>
#include <thrust/memory.h>
#include <thrust/pair.h>

struct my_tag : thrust::device_system_tag {};

template<typename T>
  thrust::pair<thrust::pointer<T,my_tag>, std::ptrdiff_t>
    get_temporary_buffer(my_tag, size_t n)
{
  // communicate that my version of get_temporary_buffer
  // was correctly dispatched
  throw my_tag();
}

void TestGetTemporaryBufferDispatchImplicit()
{
  bool correctly_dispatched = false;

  try
  {
    thrust::device_vector<int> vec(2);

    // call something we know will invoke get_temporary_buffer
    thrust::sort(thrust::retag<my_tag>(vec.begin()),
                 thrust::retag<my_tag>(vec.end()));
  }
  catch(my_tag)
  {
    correctly_dispatched = true;
  }

  ASSERT_EQUAL(true, correctly_dispatched);
}
DECLARE_UNITTEST(TestGetTemporaryBufferDispatchImplicit);

