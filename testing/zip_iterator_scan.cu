#include <unittest/unittest.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>

using namespace unittest;
using namespace thrust;


template<typename Tuple>
struct TuplePlus
{
  __host__ __device__
  Tuple operator()(Tuple x, Tuple y) const
  {
    return make_tuple(get<0>(x) + get<0>(y),
                      get<1>(x) + get<1>(y));
  }
}; // end SumTuple


template <typename T>
struct TestZipIteratorScan
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_data0 = unittest::random_samples<T>(n);
    thrust::host_vector<T> h_data1 = unittest::random_samples<T>(n);

    thrust::device_vector<T> d_data0 = h_data0;
    thrust::device_vector<T> d_data1 = h_data1;

    typedef tuple<T,T> Tuple;

    thrust::host_vector<Tuple>   h_result(n);
    thrust::device_vector<Tuple> d_result(n);

    // inclusive_scan (tuple output)
    thrust::inclusive_scan( make_zip_iterator(make_tuple(h_data0.begin(), h_data1.begin())),
                            make_zip_iterator(make_tuple(h_data0.end(),   h_data1.end())),
                            h_result.begin(),
                            TuplePlus<Tuple>());
    thrust::inclusive_scan( make_zip_iterator(make_tuple(d_data0.begin(), d_data1.begin())),
                            make_zip_iterator(make_tuple(d_data0.end(),   d_data1.end())),
                            d_result.begin(),
                            TuplePlus<Tuple>());
    ASSERT_EQUAL_QUIET(h_result, d_result);
   
    // exclusive_scan (tuple output)
    thrust::exclusive_scan( make_zip_iterator(make_tuple(h_data0.begin(), h_data1.begin())),
                            make_zip_iterator(make_tuple(h_data0.end(),   h_data1.end())),
                            h_result.begin(),
                            make_tuple<T,T>(0,0),
                            TuplePlus<Tuple>());
    thrust::exclusive_scan( make_zip_iterator(make_tuple(d_data0.begin(), d_data1.begin())),
                            make_zip_iterator(make_tuple(d_data0.end(),   d_data1.end())),
                            d_result.begin(),
                            make_tuple<T,T>(0,0),
                            TuplePlus<Tuple>());
    ASSERT_EQUAL_QUIET(h_result, d_result);

    thrust::host_vector<T>   h_result0(n);
    thrust::host_vector<T>   h_result1(n);
    thrust::device_vector<T> d_result0(n);
    thrust::device_vector<T> d_result1(n);
    
    // inclusive_scan (zip_iterator output)
    thrust::inclusive_scan( make_zip_iterator(make_tuple(h_data0.begin(), h_data1.begin())),
                            make_zip_iterator(make_tuple(h_data0.end(),   h_data1.end())),
                            make_zip_iterator(make_tuple(h_result0.begin(), h_result1.begin())),
                            TuplePlus<Tuple>());
    thrust::inclusive_scan( make_zip_iterator(make_tuple(d_data0.begin(), d_data1.begin())),
                            make_zip_iterator(make_tuple(d_data0.end(),   d_data1.end())),
                            make_zip_iterator(make_tuple(d_result0.begin(), d_result1.begin())),
                            TuplePlus<Tuple>());
    ASSERT_EQUAL_QUIET(h_result0, d_result0);
    ASSERT_EQUAL_QUIET(h_result1, d_result1);
    
    // exclusive_scan (zip_iterator output)
    thrust::exclusive_scan( make_zip_iterator(make_tuple(h_data0.begin(), h_data1.begin())),
                            make_zip_iterator(make_tuple(h_data0.end(),   h_data1.end())),
                            make_zip_iterator(make_tuple(h_result0.begin(), h_result1.begin())),
                            make_tuple<T,T>(0,0),
                            TuplePlus<Tuple>());
    thrust::exclusive_scan( make_zip_iterator(make_tuple(d_data0.begin(), d_data1.begin())),
                            make_zip_iterator(make_tuple(d_data0.end(),   d_data1.end())),
                            make_zip_iterator(make_tuple(d_result0.begin(), d_result1.begin())),
                            make_tuple<T,T>(0,0),
                            TuplePlus<Tuple>());
    ASSERT_EQUAL_QUIET(h_result0, d_result0);
    ASSERT_EQUAL_QUIET(h_result1, d_result1);
  }
};
VariableUnitTest<TestZipIteratorScan, SignedIntegralTypes> TestZipIteratorScanInstance;

