#include <unittest/unittest.h>
#include <thrust/partition.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>


template<typename Iterator1, typename Predicate, typename Iterator2>
__global__
void partition_kernel(Iterator1 first, Iterator1 last, Predicate pred, Iterator2 result)
{
  *result = thrust::partition(thrust::seq, first, last, pred);
}


template<typename T>
struct is_even
{
  __host__ __device__
  bool operator()(T x) const { return ((int) x % 2) == 0; }
};


void TestPartitionDeviceSeq(void)
{
  typedef int T;
  typedef typename thrust::device_vector<T>::iterator iterator;
  
  thrust::device_vector<T> data(5);
  data[0] = 1; 
  data[1] = 2; 
  data[2] = 1;
  data[3] = 1; 
  data[4] = 2; 

  thrust::device_vector<iterator> result(1);
  
  partition_kernel<<<1,1>>>(data.begin(), data.end(), is_even<T>(), result.begin());
  
  thrust::device_vector<T> ref(5);
  ref[0] = 2;
  ref[1] = 2;
  ref[2] = 1;
  ref[3] = 1;
  ref[4] = 1;
  
  ASSERT_EQUAL(2, (iterator)result[0] - data.begin());
  ASSERT_EQUAL(ref, data);
}
DECLARE_UNITTEST(TestPartitionDeviceSeq);


template<typename Iterator1, typename Iterator2, typename Predicate, typename Iterator3>
__global__
void partition_kernel(Iterator1 first, Iterator1 last, Iterator2 stencil_first, Predicate pred, Iterator3 result)
{
  *result = thrust::partition(thrust::seq, first, last, stencil_first, pred);
}


void TestPartitionStencilDeviceSeq()
{
  typedef int T;
  typedef typename thrust::device_vector<T>::iterator iterator;
  
  thrust::device_vector<T> data(5);
  data[0] = 0;
  data[1] = 1;
  data[2] = 0;
  data[3] = 0;
  data[4] = 1;
  
  thrust::device_vector<T> stencil(5);
  stencil[0] = 1; 
  stencil[1] = 2; 
  stencil[2] = 1;
  stencil[3] = 1; 
  stencil[4] = 2; 

  thrust::device_vector<iterator> result(1);
  
  partition_kernel<<<1,1>>>(data.begin(), data.end(), stencil.begin(), is_even<T>(), result.begin());
  
  thrust::device_vector<T> ref(5);
  ref[0] = 1;
  ref[1] = 1;
  ref[2] = 0;
  ref[3] = 0;
  ref[4] = 0;
  
  ASSERT_EQUAL(2, (iterator)result[0] - data.begin());
  ASSERT_EQUAL(ref, data);
}
DECLARE_UNITTEST(TestPartitionStencilDeviceSeq);


template<typename Iterator1, typename Iterator2, typename Iterator3, typename Predicate, typename Iterator4>
__global__
void partition_copy_kernel(Iterator1 first, Iterator1 last, Iterator2 true_result, Iterator3 false_result, Predicate pred, Iterator4 result)
{
  *result = thrust::partition_copy(thrust::seq, first, last, true_result, false_result, pred);
}


void TestPartitionCopyDeviceSeq()
{
  typedef int T;
  typedef thrust::device_vector<T>::iterator iterator;
  
  thrust::device_vector<T> data(5);
  data[0] =  1; 
  data[1] =  2; 
  data[2] =  1;
  data[3] =  1; 
  data[4] =  2; 
  
  thrust::device_vector<int> true_results(2);
  thrust::device_vector<int> false_results(3);

  typedef thrust::pair<iterator,iterator> pair_type;
  thrust::device_vector<pair_type> iterators(1);
  
  partition_copy_kernel<<<1,1>>>(data.begin(), data.end(), true_results.begin(), false_results.begin(), is_even<T>(), iterators.begin());
  
  thrust::device_vector<T> true_ref(2);
  true_ref[0] =  2;
  true_ref[1] =  2;
  
  thrust::device_vector<T> false_ref(3);
  false_ref[0] =  1;
  false_ref[1] =  1;
  false_ref[2] =  1;

  pair_type ends = iterators[0];
  
  ASSERT_EQUAL(2, ends.first - true_results.begin());
  ASSERT_EQUAL(3, ends.second - false_results.begin());
  ASSERT_EQUAL(true_ref, true_results);
  ASSERT_EQUAL(false_ref, false_results);
}
DECLARE_UNITTEST(TestPartitionCopyDeviceSeq);


template<typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Predicate, typename Iterator5>
__global__
void partition_copy_kernel(Iterator1 first, Iterator1 last, Iterator2 stencil_first, Iterator3 true_result, Iterator4 false_result, Predicate pred, Iterator5 result)
{
  *result = thrust::partition_copy(thrust::seq, first, last, stencil_first, true_result, false_result, pred);
}


template<typename Vector>
void TestPartitionCopyStencilDeviceSeq()
{
  typedef int T;
  
  thrust::device_vector<int> data(5);
  data[0] =  0; 
  data[1] =  1; 
  data[2] =  0;
  data[3] =  0; 
  data[4] =  1; 
  
  thrust::device_vector<int> stencil(5);
  stencil[0] =  1; 
  stencil[1] =  2; 
  stencil[2] =  1;
  stencil[3] =  1; 
  stencil[4] =  2; 
  
  thrust::device_vector<int> true_results(2);
  thrust::device_vector<int> false_results(3);

  typedef typename thrust::device_vector<int>::iterator iterator;
  typedef thrust::pair<iterator,iterator> pair_type;
  thrust::device_vector<pair_type> iterators(1);

  partition_copy_kernel<<<1,1>>>(data.begin(), data.end(), stencil.begin(), true_results.begin(), false_results.begin(), is_even<T>(), iterators.begin());

  pair_type ends = iterators[0];
  
  thrust::device_vector<int> true_ref(2);
  true_ref[0] =  1;
  true_ref[1] =  1;
  
  thrust::device_vector<int> false_ref(3);
  false_ref[0] =  0;
  false_ref[1] =  0;
  false_ref[2] =  0;
  
  ASSERT_EQUAL(2, ends.first - true_results.begin());
  ASSERT_EQUAL(3, ends.second - false_results.begin());
  ASSERT_EQUAL(true_ref, true_results);
  ASSERT_EQUAL(false_ref, false_results);
}
DECLARE_VECTOR_UNITTEST(TestPartitionCopyStencilDeviceSeq);


template<typename Iterator1, typename Predicate, typename Iterator2, typename Iterator3>
__global__
void stable_partition_kernel(Iterator1 first, Iterator1 last, Predicate pred, Iterator2 result, Iterator3 is_supported)
{
#if (__CUDA_ARCH__ >= 200)
  *is_supported = true;
  *result = thrust::stable_partition(thrust::seq, first, last, pred);
#else
  *is_supported = false;
#endif
}


void TestStablePartitionDeviceSeq(void)
{
  typedef int T;
  typedef typename thrust::device_vector<T>::iterator iterator;
  
  thrust::device_vector<T> data(5);
  data[0] = 1; 
  data[1] = 2; 
  data[2] = 1;
  data[3] = 1; 
  data[4] = 2; 

  thrust::device_vector<iterator> result(1);
  thrust::device_vector<bool> is_supported(1);
  
  stable_partition_kernel<<<1,1>>>(data.begin(), data.end(), is_even<T>(), result.begin(), is_supported.begin());
  
  if(is_supported[0])
  {
    thrust::device_vector<T> ref(5);
    ref[0] = 2;
    ref[1] = 2;
    ref[2] = 1;
    ref[3] = 1;
    ref[4] = 1;
    
    ASSERT_EQUAL(2, (iterator)result[0] - data.begin());
    ASSERT_EQUAL(ref, data);
  }
}
DECLARE_UNITTEST(TestStablePartitionDeviceSeq);


template<typename Iterator1, typename Iterator2, typename Predicate, typename Iterator3, typename Iterator4>
__global__
void stable_partition_kernel(Iterator1 first, Iterator1 last, Iterator2 stencil_first, Predicate pred, Iterator3 result, Iterator4 is_supported)
{
#if (__CUDA_ARCH__ >= 200)
  *is_supported = true;
  *result = thrust::stable_partition(thrust::seq, first, last, stencil_first, pred);
#else
  *is_supported = false;
#endif
}


void TestStablePartitionStencilDeviceSeq()
{
  typedef int T;
  typedef typename thrust::device_vector<T>::iterator iterator;
  
  thrust::device_vector<T> data(5);
  data[0] = 0;
  data[1] = 1;
  data[2] = 0;
  data[3] = 0;
  data[4] = 1;
  
  thrust::device_vector<T> stencil(5);
  stencil[0] = 1; 
  stencil[1] = 2; 
  stencil[2] = 1;
  stencil[3] = 1; 
  stencil[4] = 2; 

  thrust::device_vector<iterator> result(1);
  thrust::device_vector<bool> is_supported(1);
  
  stable_partition_kernel<<<1,1>>>(data.begin(), data.end(), stencil.begin(), is_even<T>(), result.begin(), is_supported.begin());
  
  if(is_supported[0])
  {
    thrust::device_vector<T> ref(5);
    ref[0] = 1;
    ref[1] = 1;
    ref[2] = 0;
    ref[3] = 0;
    ref[4] = 0;
    
    ASSERT_EQUAL(2, (iterator)result[0] - data.begin());
    ASSERT_EQUAL(ref, data);
  }
}
DECLARE_UNITTEST(TestStablePartitionStencilDeviceSeq);


template<typename Iterator1, typename Iterator2, typename Iterator3, typename Predicate, typename Iterator4>
__global__
void stable_partition_copy_kernel(Iterator1 first, Iterator1 last, Iterator2 true_result, Iterator3 false_result, Predicate pred, Iterator4 result)
{
  *result = thrust::stable_partition_copy(thrust::seq, first, last, true_result, false_result, pred);
}


void TestStablePartitionCopyDeviceSeq()
{
  typedef int T;
  typedef thrust::device_vector<T>::iterator iterator;
  
  thrust::device_vector<T> data(5);
  data[0] =  1; 
  data[1] =  2; 
  data[2] =  1;
  data[3] =  1; 
  data[4] =  2; 
  
  thrust::device_vector<int> true_results(2);
  thrust::device_vector<int> false_results(3);

  typedef thrust::pair<iterator,iterator> pair_type;
  thrust::device_vector<pair_type> iterators(1);
  
  stable_partition_copy_kernel<<<1,1>>>(data.begin(), data.end(), true_results.begin(), false_results.begin(), is_even<T>(), iterators.begin());
  
  thrust::device_vector<T> true_ref(2);
  true_ref[0] =  2;
  true_ref[1] =  2;
  
  thrust::device_vector<T> false_ref(3);
  false_ref[0] =  1;
  false_ref[1] =  1;
  false_ref[2] =  1;

  pair_type ends = iterators[0];
  
  ASSERT_EQUAL(2, ends.first - true_results.begin());
  ASSERT_EQUAL(3, ends.second - false_results.begin());
  ASSERT_EQUAL(true_ref, true_results);
  ASSERT_EQUAL(false_ref, false_results);
}
DECLARE_UNITTEST(TestStablePartitionCopyDeviceSeq);


template<typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Predicate, typename Iterator5>
__global__
void stable_partition_copy_kernel(Iterator1 first, Iterator1 last, Iterator2 stencil_first, Iterator3 true_result, Iterator4 false_result, Predicate pred, Iterator5 result)
{
  *result = thrust::stable_partition_copy(thrust::seq, first, last, stencil_first, true_result, false_result, pred);
}


template<typename Vector>
void TestStablePartitionCopyStencilDeviceSeq()
{
  typedef int T;
  
  thrust::device_vector<int> data(5);
  data[0] =  0; 
  data[1] =  1; 
  data[2] =  0;
  data[3] =  0; 
  data[4] =  1; 
  
  thrust::device_vector<int> stencil(5);
  stencil[0] =  1; 
  stencil[1] =  2; 
  stencil[2] =  1;
  stencil[3] =  1; 
  stencil[4] =  2; 
  
  thrust::device_vector<int> true_results(2);
  thrust::device_vector<int> false_results(3);

  typedef typename thrust::device_vector<int>::iterator iterator;
  typedef thrust::pair<iterator,iterator> pair_type;
  thrust::device_vector<pair_type> iterators(1);

  stable_partition_copy_kernel<<<1,1>>>(data.begin(), data.end(), stencil.begin(), true_results.begin(), false_results.begin(), is_even<T>(), iterators.begin());

  pair_type ends = iterators[0];
  
  thrust::device_vector<int> true_ref(2);
  true_ref[0] =  1;
  true_ref[1] =  1;
  
  thrust::device_vector<int> false_ref(3);
  false_ref[0] =  0;
  false_ref[1] =  0;
  false_ref[2] =  0;
  
  ASSERT_EQUAL(2, ends.first - true_results.begin());
  ASSERT_EQUAL(3, ends.second - false_results.begin());
  ASSERT_EQUAL(true_ref, true_results);
  ASSERT_EQUAL(false_ref, false_results);
}
DECLARE_VECTOR_UNITTEST(TestStablePartitionCopyStencilDeviceSeq);


void TestPartitionCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;
  typedef typename Vector::iterator   Iterator;
  
  Vector data(5);
  data[0] = 1; 
  data[1] = 2; 
  data[2] = 1;
  data[3] = 1; 
  data[4] = 2; 

  cudaStream_t s;
  cudaStreamCreate(&s);
  
  Iterator iter = thrust::partition(thrust::cuda::par(s), data.begin(), data.end(), is_even<T>());
  
  Vector ref(5);
  ref[0] = 2;
  ref[1] = 2;
  ref[2] = 1;
  ref[3] = 1;
  ref[4] = 1;
  
  ASSERT_EQUAL(iter - data.begin(), 2);
  ASSERT_EQUAL(data, ref);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestPartitionCudaStreams);

