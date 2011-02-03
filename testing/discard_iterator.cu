#include <unittest/unittest.h>
#include <thrust/iterator/discard_iterator.h>

using namespace thrust;

void TestDiscardIteratorIncrement(void)
{
  discard_iterator<> lhs(0);
  discard_iterator<> rhs(0);

  ASSERT_EQUAL(0, lhs - rhs);

  lhs++;

  ASSERT_EQUAL(1, lhs - rhs);
  
  lhs++;
  lhs++;
  
  ASSERT_EQUAL(3, lhs - rhs);

  lhs += 5;
  
  ASSERT_EQUAL(8, lhs - rhs);

  lhs -= 10;
  
  ASSERT_EQUAL(-2, lhs - rhs);
}
DECLARE_UNITTEST(TestDiscardIteratorIncrement);

void TestDiscardIteratorComparison(void)
{
  discard_iterator<> iter1(0);
  discard_iterator<> iter2(0);

  ASSERT_EQUAL(0, iter1 - iter2);
  ASSERT_EQUAL(true, iter1 == iter2);

  iter1++;
  
  ASSERT_EQUAL(1, iter1 - iter2);
  ASSERT_EQUAL(false, iter1 == iter2);
  
  iter2++;

  ASSERT_EQUAL(0, iter1 - iter2);
  ASSERT_EQUAL(true, iter1 == iter2);
  
  iter1 += 100;
  iter2 += 100;

  ASSERT_EQUAL(0, iter1 - iter2);
  ASSERT_EQUAL(true, iter1 == iter2);
}
DECLARE_UNITTEST(TestDiscardIteratorComparison);

void TestMakeDiscardIterator(void)
{
  discard_iterator<> iter0 = make_discard_iterator(13);

  *iter0 = 7;

  discard_iterator<> iter1 = make_discard_iterator(7);

  *iter1 = 13;

  ASSERT_EQUAL(6, iter0 - iter1);
}
DECLARE_UNITTEST(TestMakeDiscardIterator);

void TestZippedDiscardIterator(void)
{
  // this test is just to make sure this can compile
  typedef tuple<discard_iterator<> >  IteratorTuple1;
  typedef zip_iterator<IteratorTuple1> ZipIterator1;

  IteratorTuple1 t = thrust::make_tuple(thrust::make_discard_iterator());

  ZipIterator1 z_iter = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_discard_iterator()));

  typedef tuple<int *, discard_iterator<> > IteratorTuple2;
  typedef zip_iterator<IteratorTuple2>      ZipIterator2;

  ZipIterator2 z_iter_first = thrust::make_zip_iterator(thrust::make_tuple((int*)0, thrust::make_discard_iterator()));
  ZipIterator2 z_iter_last  = z_iter_first + 10;

  for(; z_iter_first != z_iter_last; ++z_iter_first)
  {
    ;
  }

  ASSERT_EQUAL(10, thrust::get<1>(z_iter_first.get_iterator_tuple()) - thrust::make_discard_iterator());
}
DECLARE_UNITTEST(TestZippedDiscardIterator);

