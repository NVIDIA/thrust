#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>

namespace thrust
{

template <typename OutputIterator, typename UnaryFunction>
  class transform_output_iterator;

namespace detail 
{

template <typename UnaryFunction, typename OutputIterator>
  class transform_output_iterator_proxy
{

  public:
    __host__ __device__
    transform_output_iterator_proxy(const OutputIterator& out, UnaryFunction fun) : fun(fun), out(out)
    {
    }

    template <typename T>
    __host__ __device__
    transform_output_iterator_proxy operator=(const T& x)
    {
      *out = fun(x);
      return *this;
    }

  private:
    OutputIterator out;
    UnaryFunction fun;
};

// Compute the iterator_adaptor instantiation to be used for transform_output_iterator
template <typename UnaryFunction, typename OutputIterator>
struct transform_output_iterator_base
{
    typedef thrust::iterator_adaptor
    <
        transform_output_iterator<UnaryFunction, OutputIterator>
      , OutputIterator
      , thrust::use_default
      , thrust::use_default
      , thrust::use_default
      , transform_output_iterator_proxy<UnaryFunction, OutputIterator>
    > type;
};

} // end detail
} // end thrust

