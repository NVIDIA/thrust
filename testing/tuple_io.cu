#include <unittest/unittest.h>
#include <thrust/tuple.h>
#include <sstream>

using namespace unittest;

void TestTupleIO(void)
{
  int a = 7;
  int b = 13;
  int c = 42;

  int x = 77;
  int y = 1313;
  int z = 4242;

  thrust::tuple<int,int,int> t1(a,b,c);
  thrust::tuple<int,int,int> t2(x,y,z);

  std::stringstream i;
  i << t1 << " ";
  i << thrust::set_open('[');
  i << thrust::set_delimiter(',');
  i << thrust::set_close(']');
  i << t2;
  i << thrust::set_open('(');
  i << thrust::set_delimiter(' ');
  i << thrust::set_close(')');
  i << std::endl;

  std::string t1_t2(i.str());

  ASSERT_EQUAL(t1_t2, "(7 13 42) [77,1313,4242]\n");

  thrust::tuple<int,int,int> t3;
  thrust::tuple<int,int,int> t4;

  i >> t3;  
  i << thrust::set_open('[');
  i << thrust::set_delimiter(',');
  i << thrust::set_close(']');
  i >> t4;

  ASSERT_EQUAL(t1, t3);
  ASSERT_EQUAL(t2, t4);
}
DECLARE_UNITTEST(TestTupleIO);
