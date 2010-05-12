#include <unittest/unittest.h>
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
    
    thrust::host_vector<T> h_input = unittest::random_integers<T>(n);

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


template <size_t N> __host__ __device__ void f   (int * x) { int temp = *x; f<N - 1>(x + 1); *x = temp;};
template <>         __host__ __device__ void f<0>(int * x) { }
template <size_t N>
struct CopyFunctorWithManyRegisters
{
    __host__ __device__
    void operator()(int * ptr)
    {
        f<N>(ptr);
    }
};
void TestForEachLargeRegisterFootprint()
{
    const size_t N = 100;

    thrust::device_vector<int> data(N, 12345);

    thrust::device_vector<int *> input(1, get_pointer(&data[0])); // length is irrelevant
    
    thrust::for_each(input.begin(), input.end(), CopyFunctorWithManyRegisters<N>());
}
DECLARE_UNITTEST(TestForEachLargeRegisterFootprint);


template <typename T, unsigned int N>
struct SetFixedVectorToConstant
{
    FixedVector<T,N> exemplar;

    SetFixedVectorToConstant(T scalar) : exemplar(scalar) {} 

    __host__ __device__
    void operator()(FixedVector<T,N>& t)
    {
        t = exemplar;
    }
};
template <typename T, unsigned int N>
void _TestForEachWithLargeTypes(void)
{
    size_t n = (64 * 1024) / sizeof(FixedVector<T,N>);

    thrust::host_vector< FixedVector<T,N> > h_data(n);

    for(size_t i = 0; i < h_data.size(); i++)
        h_data[i] = FixedVector<T,N>(i);

    thrust::device_vector< FixedVector<T,N> > d_data = h_data;
   
    SetFixedVectorToConstant<T,N> func(123);

    thrust::for_each(h_data.begin(), h_data.end(), func);
    thrust::for_each(d_data.begin(), d_data.end(), func);

    ASSERT_EQUAL_QUIET(h_data, d_data);
}
void TestForEachWithLargeTypes(void)
{
    _TestForEachWithLargeTypes<int,    1>();
    _TestForEachWithLargeTypes<int,    2>();
    _TestForEachWithLargeTypes<int,    4>();
    _TestForEachWithLargeTypes<int,    8>();
    _TestForEachWithLargeTypes<int,   16>();

    KNOWN_FAILURE;

    //_TestForEachWithLargeTypes<int,   32>();  // fails on Linux 32 w/ gcc 4.1
    //_TestForEachWithLargeTypes<int,   64>();
    //_TestForEachWithLargeTypes<int,  128>();
    //_TestForEachWithLargeTypes<int,  256>();
    //_TestForEachWithLargeTypes<int,  512>();
    //_TestForEachWithLargeTypes<int, 1024>();  // fails on Vista 64 w/ VS2008
}
DECLARE_UNITTEST(TestForEachWithLargeTypes);

