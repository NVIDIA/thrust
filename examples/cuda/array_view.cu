#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <iostream>

template<class RandomAccessIterator>
class array_view
{
public:
  typedef RandomAccessIterator iterator;

private:
  const iterator first;
  const iterator last;
  typedef typename thrust::iterator_traits<iterator>::difference_type size_type;
  typedef typename thrust::iterator_traits<iterator>::reference reference;


public:
  __host__ __device__
  array_view(RandomAccessIterator first, RandomAccessIterator last)
      : first(first), last(last) {}
  __host__ __device__
  ~array_view() {}

  __host__ __device__
  size_type size() const { return thrust::distance(first, last); }

  __host__ __device__
  reference operator[](size_type n)
  {
    return *(first + n);
  }
  __host__ __device__
  const reference operator[](size_type n) const
  {
    return *(first + n);
  }

  __host__ __device__
  iterator begin() 
  {
    return first;
  }
  __host__ __device__
  const iterator cbegin() const
  {
    return first;
  }
  __host__ __device__
  iterator end() 
  {
    return last;
  }
  __host__ __device__
  const iterator cend() const
  {
    return last;
  }


  __host__ __device__
  thrust::reverse_iterator<iterator> rbegin()
  {
    return thrust::reverse_iterator<iterator>(end());
  }
  __host__ __device__
  const thrust::reverse_iterator<const iterator> crbegin() const 
  {
    return thrust::reverse_iterator<const iterator>(cend());
  }
  __host__ __device__
  thrust::reverse_iterator<iterator> rend()
  {
    return thrust::reverse_iterator<iterator>(begin());
  }
  __host__ __device__
  const thrust::reverse_iterator<const iterator> crend() const 
  {
    return thrust::reverse_iterator<const iterator>(cbegin());
  }
  __host__ __device__
  reference front() 
  {
    return *begin();
  }
  __host__ __device__
  const reference front()  const
  {
    return *cbegin();
  }

  __host__ __device__
  reference back() 
  {
    return *end();
  }
  __host__ __device__
  const reference back()  const
  {
    return *cend();
  }

  __host__ __device__
  bool empty() const 
  {
    return size() == 0;
  }

};

template <class RandomAccessIterator, class Size>
array_view<RandomAccessIterator>
__host__ __device__
make_array_view(RandomAccessIterator first, Size n)
{
  return array_view<RandomAccessIterator>(first, first+n);
}

template <class RandomAccessIterator>
array_view<RandomAccessIterator>
__host__ __device__
make_array_view(RandomAccessIterator first, RandomAccessIterator last)
{
  return array_view<RandomAccessIterator>(first, last);
}

template<class ArrayView>
struct saxpy_functor : public thrust::unary_function<int,void>
{
  const float a;
  ArrayView x;
  ArrayView y;
  ArrayView z;

  __host__ __device__
  saxpy_functor(float _a, ArrayView _x, ArrayView _y, ArrayView _z)
      : a(_a), x(_x), y(_y), z(_z)
  {
  }

  __host__ __device__ 
  void operator()(int i) 
  {
    z[i] = a * x[i] + y[i];
  }
};

template<class ArrayView>
__host__ __device__
void saxpy(float A, ArrayView X, ArrayView Y, ArrayView Z)
{
  // Z = A * X + Y
  const int size = X.size();
  thrust::for_each(thrust::device,
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(size),
      saxpy_functor<ArrayView>(A,X,Y,Z));
}

template<class ArrayView>
__global__
void saxpy_kernel(float A, ArrayView X, ArrayView Y, ArrayView Z)
{
  saxpy(A, X, Y, Z);
}

int main(int argc, char* argv[])
{
  using std::cout;
  using std::endl;

  // initialize host arrays
  float x[4] = {1.0, 1.0, 1.0, 1.0};
  float y[4] = {1.0, 2.0, 3.0, 4.0};
  float z[4] = {0.0};

  thrust::device_vector<float> X(x, x + 4);
  thrust::device_vector<float> Y(y, y + 4);
  thrust::device_vector<float> Z(z, z + 4);

  saxpy_kernel<<<1, 1>>>(
      2.0, 
      make_array_view(X.begin(), X.end()),
      make_array_view(Y.begin(), thrust::distance(Y.begin(), Y.end())),
      make_array_view(Z.begin(), 4)
      );
  assert(cudaSuccess == cudaDeviceSynchronize());

  for (int i = 0, n = Z.size(); i < n; ++i)
  {
    cout << "z[" << i << "]= " << Z[i] << endl;
  }


  return 0;
}

