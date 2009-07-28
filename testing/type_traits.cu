#include <thrusttest/unittest.h>

#include <thrust/detail/type_traits.h>
#include <thrust/device_ptr.h>

struct non_pod
{ int x; int y; };

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

void TestIsNormalIterator(void)
{
    typedef typename thrust::host_vector<int> HostVector;
    typedef typename thrust::device_vector<int> DeviceVector;
    
    ASSERT_EQUAL((bool) thrust::detail::is_normal_iterator< int * >::value, true);
    ASSERT_EQUAL((bool) thrust::detail::is_normal_iterator< thrust::device_ptr<int> >::value, true);


    ASSERT_EQUAL((bool) thrust::detail::is_normal_iterator<typename HostVector::iterator>::value, true);
    ASSERT_EQUAL((bool) thrust::detail::is_normal_iterator<typename HostVector::const_iterator>::value, true);

    ASSERT_EQUAL((bool) thrust::detail::is_normal_iterator<typename DeviceVector::iterator>::value, true);
    ASSERT_EQUAL((bool) thrust::detail::is_normal_iterator<typename DeviceVector::const_iterator>::value, true);

    ASSERT_EQUAL((bool) thrust::detail::is_normal_iterator< thrust::device_ptr<int> >::value, true);

    //XXX add counting_iterator, etc
}
DECLARE_UNITTEST(TestIsNormalIterator);

