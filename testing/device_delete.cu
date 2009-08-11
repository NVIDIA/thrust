#include <thrusttest/unittest.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_new.h>
#include <thrust/device_delete.h>

struct Foo
{
  __device__
  ~Foo(void)
  {
    *set_me_upon_destruction = true;
  }

  bool *set_me_upon_destruction;
};

void TestDeviceDeleteDestructorInvocation(void)
{
  thrust::device_vector<bool> destructor_flag(1, false);

  thrust::device_ptr<Foo> foo_ptr  = thrust::device_new<Foo>();

  Foo exemplar;
  exemplar.set_me_upon_destruction = thrust::raw_pointer_cast(&destructor_flag[0]);
  *foo_ptr = exemplar;

  ASSERT_EQUAL(false, destructor_flag[0]);

  thrust::device_delete(foo_ptr);

  ASSERT_EQUAL(true, destructor_flag[0]);
}
DECLARE_UNITTEST(TestDeviceDeleteDestructorInvocation);

