/*
 *  Copyright 2008-2012 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file tuple_reduce.h
 *  \brief Reduce functions which operate on tuples.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>

namespace thrust {

namespace detail {

template<typename F, typename T>
struct tuple_reduce_impl{};

template<typename F, typename HT, typename TT>
struct tuple_reduce_impl<F, thrust::detail::cons<HT, TT> > {
    typedef thrust::detail::cons<HT, TT> input_cons;
    typedef typename F::result_type result_type;

    __host__ __device__
    static result_type fun(const F& f,
                           const input_cons& t,
                           const result_type& p) {
        return tuple_reduce_impl<F, TT>::fun(f,
                                             t.get_tail(),
                                             f(p, t.get_head()));
    }
};


template<typename F,
         typename T0,
         typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5,
         typename T6,
         typename T7,
         typename T8,
         typename T9>
struct tuple_reduce_impl<F,
                         thrust::tuple<T0, T1, T2, T3, T4,
                                       T5, T6, T7, T8, T9> > {
    typedef thrust::tuple<T0, T1, T2, T3, T4,
                          T5, T6, T7, T8, T9> input_tuple;
    typedef typename F::result_type result_type;

    __host__ __device__
    static result_type fun(
        const F& f,
        const input_tuple& t,
        const result_type& p) {
        return tuple_reduce_impl<F,
                                 thrust::detail::cons<
                                     typename input_tuple::head_type,
                                     typename input_tuple::tail_type> >
            ::fun(f, t, p);
    }
};

template<typename F>
struct tuple_reduce_impl<F,
                         thrust::null_type> {
    typedef typename F::result_type result_type;

    __host__ __device__
    static result_type fun(const F&f, thrust::null_type, const result_type& p) {
        return p;
    }
};

}

/*! \p tuple_reduce allows you to reduce a tuple to a single value.
 *  \tparam F is a Binary Function type
 *  \tparam T is a <tt>thrust::tuple</tt> type
 *  \param f is the Binary Function which does the reduction
 *  \param t is the tuple being reduced
 *  \param p is the prefix value, which is returned directly for empty
 *  tuples.  Otherwise, it is an extra element to be reduced.
 */

template<typename F,
         typename T>
__host__ __device__
typename F::result_type tuple_reduce(const F& f,
                                     const T& t,
                                     const typename F::result_type& p) {
    return detail::tuple_reduce_impl<F, T>::fun(f, t, p);
}

/*! \p tuple_sum returns the arithmetic sum of all elements of a tuple.
 *  \param t is the tuple being reduced
 *  \param p is the prefix value, which is returned directly for empty
 *  tuples.  Otherwise, it is an extra element to be reduced.
 */

template<typename T, typename P>
__host__ __device__
P tuple_sum(const T& t,
            const P& p) {
    return tuple_reduce(thrust::plus<P>(), t, p);
}

/*! \p tuple_max returns the arithmetic max of all elements of a tuple.
 *  \param t is the tuple being reduced
 *  \param p is the prefix value, which is returned directly for empty
 *  tuples.  Otherwise, it is an extra element to be reduced.
 */

template<typename T, typename P>
__host__ __device__
P tuple_max(const T& t,
            const P& p) {
    return tuple_reduce(thrust::maximum<P>(), t, p);
}

/*! \p tuple_min returns the arithmetic min of all elements of a tuple.
 *  \param t is the tuple being reduced
 *  \param p is the prefix value, which is returned directly for empty
 *  tuples.  Otherwise, it is an extra element to be reduced.
 */

template<typename T, typename P>
__host__ __device__
P tuple_min(const T& t,
                     const P& p) {
    return tuple_reduce(thrust::minimum<P>(), t, p);
}

}
