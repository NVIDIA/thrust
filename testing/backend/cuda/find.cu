#include <unittest/unittest.h>
#include <thrust/find.h>
#include <thrust/execution_policy.h>


template<typename T>
struct equal_to_value_pred
{
    T value;

    equal_to_value_pred(T value) : value(value) {}

    __host__ __device__
    bool operator()(T v) const { return v == value; }
};


template<typename T>
struct not_equal_to_value_pred
{
    T value;

    not_equal_to_value_pred(T value) : value(value) {}

    __host__ __device__
    bool operator()(T v) const { return v != value; }
};


template<typename T>
struct less_than_value_pred
{
    T value;

    less_than_value_pred(T value) : value(value) {}

    __host__ __device__
    bool operator()(T v) const { return v < value; }
};


template<typename Iterator, typename T, typename Iterator2>
__global__ void find_kernel(Iterator first, Iterator last, T value, Iterator2 result)
{
  *result = thrust::find(thrust::seq, first, last, value);
}


template<typename T>
struct TestFindDeviceSeq
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;
    
    typename thrust::host_vector<T>::iterator   h_iter;

    typedef typename thrust::device_vector<T>::iterator iter_type;
    thrust::device_vector<iter_type> d_result(1);
    
    h_iter = thrust::find(h_data.begin(), h_data.end(), T(0));
    find_kernel<<<1,1>>>(d_data.begin(), d_data.end(), T(0), d_result.begin());

    ASSERT_EQUAL(h_iter - h_data.begin(), (iter_type)d_result[0] - d_data.begin());
    
    for(size_t i = 1; i < n; i *= 2)
    {
      T sample = h_data[i];
      h_iter = thrust::find(h_data.begin(), h_data.end(), sample);
      find_kernel<<<1,1>>>(d_data.begin(), d_data.end(), sample, d_result.begin());
      ASSERT_EQUAL(h_iter - h_data.begin(), (iter_type)d_result[0] - d_data.begin());
    }
  }
};
VariableUnitTest<TestFindDeviceSeq, SignedIntegralTypes> TestFindDeviceSeqInstance;


template<typename Iterator, typename Predicate, typename Iterator2>
__global__ void find_if_kernel(Iterator first, Iterator last, Predicate pred, Iterator2 result)
{
  *result = thrust::find_if(thrust::seq, first, last, pred);
}


template<typename T>
struct TestFindIfDeviceSeq
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;
    
    typename thrust::host_vector<T>::iterator   h_iter;

    typedef typename thrust::device_vector<T>::iterator iter_type;
    thrust::device_vector<iter_type> d_result(1);
    
    h_iter = thrust::find_if(h_data.begin(), h_data.end(), equal_to_value_pred<T>(0));
    find_if_kernel<<<1,1>>>(d_data.begin(), d_data.end(), equal_to_value_pred<T>(0), d_result.begin());
    ASSERT_EQUAL(h_iter - h_data.begin(), (iter_type)d_result[0] - d_data.begin());
    
    for (size_t i = 1; i < n; i *= 2)
    {
      T sample = h_data[i];
      h_iter = thrust::find_if(h_data.begin(), h_data.end(), equal_to_value_pred<T>(sample));
      find_if_kernel<<<1,1>>>(d_data.begin(), d_data.end(), equal_to_value_pred<T>(sample), d_result.begin());
      ASSERT_EQUAL(h_iter - h_data.begin(), (iter_type)d_result[0] - d_data.begin());
    }
  }
};
VariableUnitTest<TestFindIfDeviceSeq, SignedIntegralTypes> TestFindIfDeviceSeqInstance;


template<typename Iterator, typename Predicate, typename Iterator2>
__global__ void find_if_not_kernel(Iterator first, Iterator last, Predicate pred, Iterator2 result)
{
  *result = thrust::find_if_not(thrust::seq, first, last, pred);
}


template<typename T>
struct TestFindIfNotDeviceSeq
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;
    
    typename thrust::host_vector<T>::iterator   h_iter;

    typedef typename thrust::device_vector<T>::iterator iter_type;
    thrust::device_vector<iter_type> d_result(1);
    
    h_iter = thrust::find_if_not(h_data.begin(), h_data.end(), not_equal_to_value_pred<T>(0));
    find_if_not_kernel<<<1,1>>>(d_data.begin(), d_data.end(), not_equal_to_value_pred<T>(0), d_result.begin());
    ASSERT_EQUAL(h_iter - h_data.begin(), (iter_type)d_result[0] - d_data.begin());
    
    for(size_t i = 1; i < n; i *= 2)
    {
      T sample = h_data[i];
      h_iter = thrust::find_if_not(h_data.begin(), h_data.end(), not_equal_to_value_pred<T>(sample));
      find_if_not_kernel<<<1,1>>>(d_data.begin(), d_data.end(), not_equal_to_value_pred<T>(sample), d_result.begin());
      ASSERT_EQUAL(h_iter - h_data.begin(), (iter_type)d_result[0] - d_data.begin());
    }
  }
};
VariableUnitTest<TestFindIfNotDeviceSeq, SignedIntegralTypes> TestFindIfNotDeviceSeqInstance;

