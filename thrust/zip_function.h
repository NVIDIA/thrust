#pragma once

#include <type_traits>
#include <thrust/tuple.h>

namespace thrust
{

namespace detail
{
// implementation of C++14's std::index_sequence
template <std::size_t... Is>
struct index_sequence {};

template <std::size_t N, std::size_t... Is>
struct make_index_sequence
: make_index_sequence<N-1, N-1, Is...> {};

template <std::size_t... Is>
struct make_index_sequence<0, Is...> : index_sequence<Is...> {};

} // end detail

template <typename Function>
class zip_function
{

private:

    Function fun;

    template <typename Tuple, std::size_t... Is>
    __host__ __device__
    auto call_fun(Tuple&& t, detail::index_sequence<Is...>) -> decltype(fun(thrust::get<Is>(std::forward<Tuple>(t))...))
    {
        return fun(thrust::get<Is>(std::forward<Tuple>(t))...);
    }

    template <typename Tuple>
    using index_sequence_for_tuple = detail::make_index_sequence<thrust::tuple_size<typename std::decay<Tuple>::type>::value>;

public:

     __host__ __device__
    zip_function(const Function& fun) : fun(fun) {}

    template <typename Tuple>
    __host__ __device__
    auto operator()(Tuple&& t) -> decltype(this->call_fun(std::forward<Tuple>(t), index_sequence_for_tuple<Tuple>()))
    {
        return call_fun(std::forward<Tuple>(t), index_sequence_for_tuple<Tuple>());
    }
}; // end zip_function

template <typename Function>
__host__ __device__
zip_function<typename std::decay<Function>::type> make_zip_function(Function&& fun)
{
    return zip_function<typename std::decay<Function>::type>(std::forward<Function>(fun));
}

} // end thrust

