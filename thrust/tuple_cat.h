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


/*! \file tuple_cat.h
 *  \brief Routines for concatenating tuples
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/tuple.h>
#include <thrust/pair.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/static_assert.h>

namespace thrust {
namespace detail {
    
template<int i, typename TT>
struct flattened_tuple_element {
    typedef typename flattened_tuple_element<i,
    thrust::detail::cons<typename TT::head_type,
    typename TT::tail_type> >::type type;
};

template<int i, typename HT, typename TT>
struct flattened_tuple_element<i, thrust::detail::cons<HT, TT> > {
    static const int HTL = thrust::tuple_size<HT>::value;
    static const bool in_HT = i < HTL;
    typedef typename eval_if<in_HT,
        thrust::tuple_element<i, HT>,
        flattened_tuple_element<i-HTL, TT> >::type type;
};

template<int i>
struct flattened_tuple_element<i, thrust::null_type> {
    typedef thrust::null_type type;
};

} //end namespace thrust::detail

template<
    typename T0=thrust::tuple<>,
    typename T1=thrust::tuple<>,
    typename T2=thrust::tuple<>,
    typename T3=thrust::tuple<>,
    typename T4=thrust::tuple<>,
    typename T5=thrust::tuple<>,
    typename T6=thrust::tuple<>,
    typename T7=thrust::tuple<>,
    typename T8=thrust::tuple<>,
    typename T9=thrust::tuple<> >
struct tuple_cat_result {
    // ========================================================================
    // X Note to the user: If you've found this line due to a compiler error, X
    // X it's because the concatenated tuple type is too long.                X
    // X Thrust tuples can have 10 elements, maximum.                         X
    // ========================================================================
    THRUST_STATIC_ASSERT(thrust::tuple_size<T0>::value +
                         thrust::tuple_size<T1>::value +
                         thrust::tuple_size<T2>::value +
                         thrust::tuple_size<T3>::value +
                         thrust::tuple_size<T4>::value +
                         thrust::tuple_size<T5>::value +
                         thrust::tuple_size<T6>::value +
                         thrust::tuple_size<T7>::value +
                         thrust::tuple_size<T8>::value +
                         thrust::tuple_size<T9>::value <= 10);

    typedef thrust::tuple<
    T0, T1, T2, T3, T4,
    T5, T6, T7, T8, T9> nested_type;

    typedef thrust::tuple<
    typename detail::flattened_tuple_element<0, nested_type>::type,
    typename detail::flattened_tuple_element<1, nested_type>::type,
    typename detail::flattened_tuple_element<2, nested_type>::type,
    typename detail::flattened_tuple_element<3, nested_type>::type,
    typename detail::flattened_tuple_element<4, nested_type>::type,
    typename detail::flattened_tuple_element<5, nested_type>::type,
    typename detail::flattened_tuple_element<6, nested_type>::type,
    typename detail::flattened_tuple_element<7, nested_type>::type,
    typename detail::flattened_tuple_element<8, nested_type>::type,
    typename detail::flattened_tuple_element<9, nested_type>::type> type;
};



namespace detail {
    
template<int i, typename TT>
struct flattened_tuple_get {
    __host__ __device__
    static typename flattened_tuple_element<i, TT>::type fun(const TT& tt) {
        typedef thrust::detail::cons<typename TT::head_type, typename TT::tail_type> cons_type;
        return flattened_tuple_get<i, cons_type>::fun(
            cons_type(
                tt.get_head(),
                tt.get_tail()));
    }
};

template<typename CT, typename RT, int i, int HTL, bool in_HT>
__host__ __device__
typename enable_if<in_HT, RT>::type flattened_tuple_extract(const CT& cs) {
    return thrust::get<i>(cs.get_head());
}

template<typename CT, typename RT, int i, int HTL, bool in_HT>
__host__ __device__
typename disable_if<in_HT, RT>::type flattened_tuple_extract(const CT& cs) {
    return flattened_tuple_get<i-HTL, typename CT::tail_type>::fun(cs.get_tail());
}

template<int i, typename HT, typename TT>
struct flattened_tuple_get<i, thrust::detail::cons<HT, TT> > {
    typedef thrust::detail::cons<HT, TT> cons_type;
    static const int HTL = thrust::tuple_size<HT>::value;
    static const bool in_HT = i < HTL;
    typedef typename flattened_tuple_element<i, cons_type>::type el_type;

    __host__ __device__
    static el_type fun(const cons_type& cs) {
        return flattened_tuple_extract<cons_type, el_type, i, HTL, in_HT>(cs);
    }
};

template<int i>
struct flattened_tuple_get<i, thrust::null_type> {
    __host__ __device__
    static thrust::null_type fun(const thrust::null_type& n) {
        return thrust::null_type();
    }
};

template<
    typename T0=thrust::null_type,
    typename T1=thrust::null_type,
    typename T2=thrust::null_type,
    typename T3=thrust::null_type,
    typename T4=thrust::null_type,
    typename T5=thrust::null_type,
    typename T6=thrust::null_type,
    typename T7=thrust::null_type,
    typename T8=thrust::null_type,
    typename T9=thrust::null_type >
struct tuple_cat_impl {
    typedef thrust::tuple<
    T0, T1, T2, T3, T4,
    T5, T6, T7, T8, T9> input_type;
    typedef typename tuple_cat_result<
    T0, T1, T2, T3, T4,
    T5, T6, T7, T8, T9>::type result_type;
    __host__ __device__
    static result_type fun(const input_type& in) {
        return result_type(
            detail::flattened_tuple_get<0, input_type>::fun(in),
            detail::flattened_tuple_get<1, input_type>::fun(in),
            detail::flattened_tuple_get<2, input_type>::fun(in),
            detail::flattened_tuple_get<3, input_type>::fun(in),
            detail::flattened_tuple_get<4, input_type>::fun(in),
            detail::flattened_tuple_get<5, input_type>::fun(in),
            detail::flattened_tuple_get<6, input_type>::fun(in),
            detail::flattened_tuple_get<7, input_type>::fun(in),
            detail::flattened_tuple_get<8, input_type>::fun(in),
            detail::flattened_tuple_get<9, input_type>::fun(in));
    }
};

} //end namespace thrust::detail

__host__ __device__
thrust::tuple<> tuple_cat() {
    return thrust::tuple<>();
}

template<typename T0>
__host__ __device__
T0 tuple_cat(const T0& t0) {
    return t0;
}

template<typename T0, typename T1>
__host__ __device__
typename tuple_cat_result<T0, T1>::type tuple_cat(const T0& t0,
                                                  const T1& t1) {
    return detail::tuple_cat_impl<T0, T1>::fun(
        thrust::tuple<T0, T1>(
            t0, t1));
}

template<typename T0, typename T1, typename T2>
__host__ __device__
typename tuple_cat_result<T0, T1, T2>::type tuple_cat(const T0& t0,
                                                      const T1& t1,
                                                      const T2& t2) {
    return detail::tuple_cat_impl<T0, T1, T2>::fun(
        thrust::tuple<T0, T1, T2>(
            t0, t1, t2));
}

template<typename T0, typename T1, typename T2, typename T3>
__host__ __device__
typename tuple_cat_result<T0, T1, T2, T3>::type tuple_cat(const T0& t0,
                                                          const T1& t1,
                                                          const T2& t2,
                                                          const T3& t3) {
    return detail::tuple_cat_impl<T0, T1, T2, T3>::fun(
        thrust::tuple<T0, T1, T2, T3>(
            t0, t1, t2, t3));
}


template<typename T0, typename T1, typename T2, typename T3, typename T4>
__host__ __device__
typename tuple_cat_result<T0, T1, T2, T3, T4>::type tuple_cat(const T0& t0,
                                                              const T1& t1,
                                                              const T2& t2,
                                                              const T3& t3,
                                                              const T4& t4) {
    return detail::tuple_cat_impl<T0, T1, T2, T3, T4>::fun(
        thrust::tuple<T0, T1, T2, T3, T4>(
            t0, t1, t2, t3, t4));
}

template<typename T0, typename T1, typename T2, typename T3, typename T4,
    typename T5>
__host__ __device__
typename tuple_cat_result<
    T0, T1, T2, T3, T4,
    T5>::type tuple_cat(const T0& t0,
                        const T1& t1,
                        const T2& t2,
                        const T3& t3,
                        const T4& t4,
                        const T5& t5) {
    return detail::tuple_cat_impl<T0, T1, T2, T3, T4, T5>::fun(
        thrust::tuple<T0, T1, T2, T3, T4, T5>(
            t0, t1, t2, t3, t4, t5));
}

template<typename T0, typename T1, typename T2, typename T3, typename T4,
    typename T5, typename T6>
__host__ __device__
typename tuple_cat_result<
    T0, T1, T2, T3, T4,
    T5, T6>::type tuple_cat(const T0& t0,
                            const T1& t1,
                            const T2& t2,
                            const T3& t3,
                            const T4& t4,
                            const T5& t5,
                            const T6& t6) {
    return detail::tuple_cat_impl<T0, T1, T2, T3, T4, T5, T6>::fun(
        thrust::tuple<T0, T1, T2, T3, T4, T5, T6>(
            t0, t1, t2, t3, t4, t5, t6));
}

template<typename T0, typename T1, typename T2, typename T3, typename T4,
    typename T5, typename T6, typename T7>
__host__ __device__
typename tuple_cat_result<
    T0, T1, T2, T3, T4,
    T5, T6, T7>::type tuple_cat(const T0& t0,
                                const T1& t1,
                                const T2& t2,
                                const T3& t3,
                                const T4& t4,
                                const T5& t5,
                                const T6& t6,
                                const T7& t7) {
    return detail::tuple_cat_impl<T0, T1, T2, T3, T4, T5, T6, T7>::fun(
        thrust::tuple<T0, T1, T2, T3, T4, T5, T6, T7>(
            t0, t1, t2, t3, t4, t5, t6, t7));
}


template<typename T0, typename T1, typename T2, typename T3, typename T4,
    typename T5, typename T6, typename T7, typename T8>
__host__ __device__
typename tuple_cat_result<
    T0, T1, T2, T3, T4,
    T5, T6, T7, T8>::type tuple_cat(const T0& t0,
                                    const T1& t1,
                                    const T2& t2,
                                    const T3& t3,
                                    const T4& t4,
                                    const T5& t5,
                                    const T6& t6,
                                    const T7& t7,
                                    const T8& t8) {
    return detail::tuple_cat_impl<T0, T1, T2, T3, T4, T5, T6, T7, T8>::fun(
        thrust::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8>(
            t0, t1, t2, t3, t4, t5, t6, t7, t8));
}

template<typename T0, typename T1, typename T2, typename T3, typename T4,
    typename T5, typename T6, typename T7, typename T8, typename T9>
__host__ __device__
typename tuple_cat_result<
    T0, T1, T2, T3, T4,
    T5, T6, T7, T8, T9>::type tuple_cat(const T0& t0,
                                        const T1& t1,
                                        const T2& t2,
                                        const T3& t3,
                                        const T4& t4,
                                        const T5& t5,
                                        const T6& t6,
                                        const T7& t7,
                                        const T8& t8,
                                        const T9& t9) {
    return detail::tuple_cat_impl<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>::fun(
        thrust::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>(
            t0, t1, t2, t3, t4, t5, t6, t7, t8, t9));
}

//Overloads for concatenating empty tuples.
__host__ __device__
thrust::tuple<> tuple_cat(
    const thrust::tuple<>& t0,
    const thrust::tuple<>& t1) {
    return thrust::tuple<>();
}

__host__ __device__
thrust::tuple<> tuple_cat(
    const thrust::tuple<>& t0,
    const thrust::tuple<>& t1,
    const thrust::tuple<>& t2) {
    return thrust::tuple<>();
}

__host__ __device__
thrust::tuple<> tuple_cat(
    const thrust::tuple<>& t0,
    const thrust::tuple<>& t1,
    const thrust::tuple<>& t2,
    const thrust::tuple<>& t3) {
    return thrust::tuple<>();
}

__host__ __device__
thrust::tuple<> tuple_cat(
    const thrust::tuple<>& t0,
    const thrust::tuple<>& t1,
    const thrust::tuple<>& t2,
    const thrust::tuple<>& t3,
    const thrust::tuple<>& t4) {
    return thrust::tuple<>();
}

__host__ __device__
thrust::tuple<> tuple_cat(
    const thrust::tuple<>& t0,
    const thrust::tuple<>& t1,
    const thrust::tuple<>& t2,
    const thrust::tuple<>& t3,
    const thrust::tuple<>& t4,
    const thrust::tuple<>& t5) {
    return thrust::tuple<>();
}

__host__ __device__
thrust::tuple<> tuple_cat(
    const thrust::tuple<>& t0,
    const thrust::tuple<>& t1,
    const thrust::tuple<>& t2,
    const thrust::tuple<>& t3,
    const thrust::tuple<>& t4,
    const thrust::tuple<>& t5,
    const thrust::tuple<>& t6) {
    return thrust::tuple<>();
}

__host__ __device__
thrust::tuple<> tuple_cat(
    const thrust::tuple<>& t0,
    const thrust::tuple<>& t1,
    const thrust::tuple<>& t2,
    const thrust::tuple<>& t3,
    const thrust::tuple<>& t4,
    const thrust::tuple<>& t5,
    const thrust::tuple<>& t6,
    const thrust::tuple<>& t7) {
    return thrust::tuple<>();
}

__host__ __device__
thrust::tuple<> tuple_cat(
    const thrust::tuple<>& t0,
    const thrust::tuple<>& t1,
    const thrust::tuple<>& t2,
    const thrust::tuple<>& t3,
    const thrust::tuple<>& t4,
    const thrust::tuple<>& t5,
    const thrust::tuple<>& t6,
    const thrust::tuple<>& t7,
    const thrust::tuple<>& t8) {
    return thrust::tuple<>();
}

__host__ __device__
thrust::tuple<> tuple_cat(
    const thrust::tuple<>& t0,
    const thrust::tuple<>& t1,
    const thrust::tuple<>& t2,
    const thrust::tuple<>& t3,
    const thrust::tuple<>& t4,
    const thrust::tuple<>& t5,
    const thrust::tuple<>& t6,
    const thrust::tuple<>& t7,
    const thrust::tuple<>& t8,
    const thrust::tuple<>& t9) {
    return thrust::tuple<>();
}

} //end namespace thrust
