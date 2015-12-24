#pragma once

#include <thrust/detail/config.h>
#include <thrust/iterator/detail/transform_output_iterator.inl>

namespace thrust
{

template <typename UnaryFunction, typename OutputIterator>
  class transform_output_iterator
    : public detail::transform_output_iterator_base<UnaryFunction, OutputIterator>::type
{

  public:

    typedef typename
    detail::transform_output_iterator_base<UnaryFunction, OutputIterator>::type
    super_t;

    friend class thrust::iterator_core_access;

    __host__ __device__
    transform_output_iterator(const OutputIterator& out, UnaryFunction fun) : super_t(out), fun(fun)
    {
    }

  private:

    __host__ __device__
    typename super_t::reference dereference() const
    {
        return detail::transform_output_iterator_proxy<UnaryFunction, OutputIterator>(this->base_reference(), fun);
    }

    UnaryFunction fun;

}; // end transform_output_iterator


template <typename UnaryFunction, typename OutputIterator>
transform_output_iterator<UnaryFunction, OutputIterator>
__host__ __device__
make_transform_output_iterator(OutputIterator out, UnaryFunction fun)
{
    return transform_output_iterator<UnaryFunction, OutputIterator>(out, fun);
} // end make_transform_output_iterator

 
} // end thrust

