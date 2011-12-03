#include <thrust/detail/type_traits/result_of.h>
#include <thrust/system/tbb/vector.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <typeinfo>
#include <vector>
#include "raw_reference_cast.h"

template<typename Function,
         typename Reference,
         typename Result = typename thrust::detail::result_of<
           Function(typename thrust::detail::raw_reference<Reference>::type)
         >::type
        >
  struct unwrap_reference_and_call1
    : Function
{
  typedef Function super_t;

  unwrap_reference_and_call1()
    : super_t()
  {}

  unwrap_reference_and_call1(const Function &f)
    : super_t(f)
  {}

  __host__ __device__
  Result operator()(Reference ref) const
  {
    // cast away constness to call possible non-const super_t::operator()
    super_t &self = const_cast<unwrap_reference_and_call1&>(*this);
    return self(thrust::raw_reference_cast(ref));
  }
};

struct print_tuple
{
  template<typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    std::cout << thrust::get<0>(t) << ", " << thrust::get<1>(t);
  }

//  template<typename Tuple>
//  __host__ __device__
//  void operator()(Tuple t)
//  {
//    int &first = thrust::get<0>(t);
//    int &second = thrust::get<1>(t);
//    std::cout << first << ", " << second;
//  }

//  template<typename Tuple>
//  __host__ __device__
//  void operator()(Tuple &t)
//  {
//#if !__CUDA_ARCH__
//    std::cout << thrust::get<0>(t) << ", " << thrust::get<1>(t);
//#endif
//  }

//  template<typename Tuple>
//  __host__ __device__
//  void operator()(const Tuple &t)
//  {
//#if !__CUDA_ARCH__
//    std::cout << thrust::get<0>(t) << ", " << thrust::get<1>(t);
//#endif
//  }

//  template<typename T1, typename T2>
//  __host__ __device__
//  void operator()(const thrust::tuple<T1,T2> &t)
//  {
//#if !__CUDA_ARCH__
//    std::cout << thrust::get<0>(t) << ", " << thrust::get<1>(t);
//#endif
//  }

//  __host__ __device__
//  void operator()(const thrust::tuple<int,int> &t)
//  {
//#if !__CUDA_ARCH__
//    std::cout << thrust::get<0>(t) << ", " << thrust::get<1>(t);
//#endif
//  }

//  __host__ __device__
//  void operator()(const thrust::tuple<const int&, const int &> &t)
//  {
//#if !__CUDA_ARCH__
//    std::cout << thrust::get<0>(t) << ", " << thrust::get<1>(t);
//#endif
//  }
};

int main()
{
  typedef thrust::tbb::vector<int> vector;
  //typedef std::vector<int> vector;
  vector int_vec(10);

  std::cout << "is_wrapped_reference: " << thrust::detail::is_wrapped_reference<vector::reference>::value << std::endl;
  std::cout << "is_wrapped_reference: " << thrust::detail::is_wrapped_reference<const vector::reference>::value << std::endl;
  std::cout << "is_wrapped_reference: " << thrust::detail::is_wrapped_reference<typename thrust::detail::remove_cv<const vector::reference>::type>::value << std::endl;

  std::cout << "raw_reference<vector::reference>:type: " << typeid(typename thrust::detail::raw_reference<vector::reference>::type).name() << std::endl;

  int &x = thrust::raw_reference_cast(int_vec.front());
  int &y = thrust::raw_reference_cast(x);

  thrust::sequence(int_vec.begin(), int_vec.end());

  typedef thrust::identity<int> function;
  typedef vector::reference     reference;

  typedef unwrap_reference_and_call1<function,reference> unwrapper;

  function f;
  unwrapper unwrap_and_call_f(f);

  for(vector::iterator i = int_vec.begin();
      i != int_vec.end();
      ++i)
  {
    std::cout << unwrap_and_call_f(*i) << std::endl;
  }

  typedef thrust::tuple<vector::iterator,vector::iterator> iterator_tuple;
  typedef thrust::zip_iterator<iterator_tuple> zip_iterator;
  typedef thrust::tuple<vector::iterator::reference,vector::iterator::reference> reference_tuple;

  zip_iterator first = thrust::make_zip_iterator(thrust::make_tuple(int_vec.begin(), int_vec.begin()));
  zip_iterator last  = thrust::make_zip_iterator(thrust::make_tuple(int_vec.end(), int_vec.end()));

  typedef unwrap_reference_and_call1<print_tuple,reference_tuple,void> unwrap_and_print;
  unwrap_and_print printer;

  for(zip_iterator i = first;
      i != last;
      ++i)
  {
    printer(*i);
    std::cout << std::endl;
  }

  return 0;
}

