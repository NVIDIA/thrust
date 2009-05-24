#include <thrusttest/unittest.h>
#include <thrust/for_each.h>
#include <thrust/device_ptr.h>

template <typename T>
class mark_present_for_each
{
    public:
        T * ptr;
        __host__ __device__ void operator()(T x){ ptr[(int) x] = 1; }
};

template <typename T>
  T *get_pointer(T *ptr)
{
  return ptr;
}

template <typename T>
  T *get_pointer(const thrust::device_ptr<T> ptr)
{
  return ptr.get();
}

template <class Vector>
void TestForEachSimple(void)
{
    typedef typename Vector::value_type T;

    Vector input(5);
    Vector output(7, (T) 0);
    
    input[0] = 3; input[1] = 2; input[2] = 3; input[3] = 4; input[4] = 6;

    mark_present_for_each<T> f;
    f.ptr = get_pointer(&output[0]);

    thrust::for_each(input.begin(), input.end(), f);

    ASSERT_EQUAL(output[0], 0);
    ASSERT_EQUAL(output[1], 0);
    ASSERT_EQUAL(output[2], 1);
    ASSERT_EQUAL(output[3], 1);
    ASSERT_EQUAL(output[4], 1);
    ASSERT_EQUAL(output[5], 0);
    ASSERT_EQUAL(output[6], 1);
}
DECLARE_VECTOR_UNITTEST(TestForEachSimple);


template <typename T>
void TestForEach(const size_t n)
{
    const size_t output_size = std::min((size_t) 10, 2 * n);
    
    thrust::host_vector<T> h_input = thrusttest::random_integers<T>(n);

    for(size_t i = 0; i < n; i++)
        h_input[i] =  ((size_t) h_input[i]) % output_size;
    
    thrust::device_vector<T> d_input = h_input;

    thrust::host_vector<T>   h_output(output_size, (T) 0);
    thrust::device_vector<T> d_output(output_size, (T) 0);

    mark_present_for_each<T> h_f;
    mark_present_for_each<T> d_f;
    h_f.ptr = &h_output[0];
    d_f.ptr = (&d_output[0]).get();
    
    thrust::for_each(h_input.begin(), h_input.end(), h_f);
    thrust::for_each(d_input.begin(), d_input.end(), d_f);

    ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestForEach);
