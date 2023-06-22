#include <unittest/unittest.h>
#include <thrust/sort.h>
#include <thrust/functional.h>


template <typename T, unsigned int N>
void _TestStableSortWithLargeKeys(void)
{
    size_t n = (128 * 1024) / sizeof(FixedVector<T,N>);

    thrust::host_vector< FixedVector<T,N> > h_keys(n);

    for(size_t i = 0; i < n; i++)
        // XXX Use proper random number generation facility.
        h_keys[i] = FixedVector<T,N>(rand());

    thrust::device_vector< FixedVector<T,N> > d_keys = h_keys;
    
    thrust::stable_sort(h_keys.begin(), h_keys.end());
    thrust::stable_sort(d_keys.begin(), d_keys.end());

    ASSERT_EQUAL_QUIET(h_keys, d_keys);
}

void TestStableSortWithLargeKeys(void)
{
    _TestStableSortWithLargeKeys<int,    2>();
    _TestStableSortWithLargeKeys<int,   17>();
    _TestStableSortWithLargeKeys<int,  128>();
}
DECLARE_UNITTEST(TestStableSortWithLargeKeys);

