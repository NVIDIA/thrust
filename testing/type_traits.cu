#include <unittest/unittest.h>

#include <thrust/detail/type_traits.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/device_ptr.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

struct non_pod
{
  // non-pods can have constructors
  non_pod(void)
  {}

  int x; int y;
};

void TestIsPlainOldData(void)
{
    // primitive types
    ASSERT_EQUAL((bool)thrust::detail::is_pod<bool>::value, true);

    ASSERT_EQUAL((bool)thrust::detail::is_pod<char>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<signed char>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<unsigned char>::value, true);
    
    ASSERT_EQUAL((bool)thrust::detail::is_pod<short>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<signed short>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<unsigned short>::value, true);

    ASSERT_EQUAL((bool)thrust::detail::is_pod<int>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<signed int>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<unsigned int>::value, true);
    
    ASSERT_EQUAL((bool)thrust::detail::is_pod<long>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<signed long>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<unsigned long>::value, true);
    
    ASSERT_EQUAL((bool)thrust::detail::is_pod<long long>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<signed long long>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<unsigned long long>::value, true);
    
    ASSERT_EQUAL((bool)thrust::detail::is_pod<float>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<double>::value, true);
    
    // void
    ASSERT_EQUAL((bool)thrust::detail::is_pod<void>::value, true);

    // structs
    ASSERT_EQUAL((bool)thrust::detail::is_pod<non_pod>::value, false);

    // pointers
    ASSERT_EQUAL((bool)thrust::detail::is_pod<char *>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<int *>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<int **>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<non_pod *>::value, true);

    // const types
    ASSERT_EQUAL((bool)thrust::detail::is_pod<const int>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<const int *>::value, true);
}
DECLARE_UNITTEST(TestIsPlainOldData);

void TestIsTrivialIterator(void)
{
    typedef typename thrust::host_vector<int>   HostVector;
    typedef typename thrust::device_vector<int> DeviceVector;
    
    ASSERT_EQUAL((bool) thrust::detail::is_trivial_iterator< int * >::value, true);
    ASSERT_EQUAL((bool) thrust::detail::is_trivial_iterator< thrust::device_ptr<int> >::value, true);


    ASSERT_EQUAL((bool) thrust::detail::is_trivial_iterator<typename HostVector::iterator>::value, true);
    ASSERT_EQUAL((bool) thrust::detail::is_trivial_iterator<typename HostVector::const_iterator>::value, true);

    ASSERT_EQUAL((bool) thrust::detail::is_trivial_iterator<typename DeviceVector::iterator>::value, true);
    ASSERT_EQUAL((bool) thrust::detail::is_trivial_iterator<typename DeviceVector::const_iterator>::value, true);

    ASSERT_EQUAL((bool) thrust::detail::is_trivial_iterator< thrust::device_ptr<int> >::value, true);

    typedef typename thrust::tuple< HostVector::iterator,   HostVector::iterator   > HostIteratorTuple;

    typedef typename thrust::constant_iterator<int> ConstantIterator;
    typedef typename thrust::counting_iterator<int> CountingIterator;
    typedef typename thrust::transform_iterator<thrust::identity<int>, HostVector::iterator > TransformIterator;
    typedef typename thrust::zip_iterator< HostIteratorTuple >  ZipIterator;

    ASSERT_EQUAL((bool) thrust::detail::is_trivial_iterator<ConstantIterator>::value,  false);
    ASSERT_EQUAL((bool) thrust::detail::is_trivial_iterator<CountingIterator>::value,  false);
    ASSERT_EQUAL((bool) thrust::detail::is_trivial_iterator<TransformIterator>::value, false);
    ASSERT_EQUAL((bool) thrust::detail::is_trivial_iterator<ZipIterator>::value,       false);

}
DECLARE_UNITTEST(TestIsTrivialIterator);

