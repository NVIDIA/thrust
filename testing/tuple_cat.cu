#include <unittest/unittest.h>
#include <thrust/tuple.h>

using namespace unittest;


void TestTupleCat(void)
{
    typedef int A;
    typedef float B;
    typedef long C;
    typedef double D;
    A a = 1;
    B b = 3.14f;
    C c = 3;
    D d = 2.78;
    
    typedef thrust::tuple<A, B> AB;
    AB ab = thrust::make_tuple(a,
                               b);
    typedef thrust::tuple<C, D> CD;
    CD cd = thrust::make_tuple(c,
                               d);
    
    
    typedef typename thrust::tuple<A,B,C,D> concat_type;

    concat_type abcd = thrust::tuple_cat(ab, cd);
    ASSERT_EQUAL(thrust::get<0>(abcd), a);
    ASSERT_EQUAL(thrust::get<1>(abcd), b);
    ASSERT_EQUAL(thrust::get<2>(abcd), c);
    ASSERT_EQUAL(thrust::get<3>(abcd), d);
    
    //Test concatenating empty tuples.
    thrust::tuple<> x;
    
    //Empty with empty
    thrust::tuple<> empty = tuple_cat(x, x);
    
    //Empty in front
    concat_type y = tuple_cat(x, abcd);
    ASSERT_EQUAL(thrust::get<0>(y), thrust::get<0>(abcd));
    ASSERT_EQUAL(thrust::get<1>(y), thrust::get<1>(abcd));
    ASSERT_EQUAL(thrust::get<2>(y), thrust::get<2>(abcd));
    ASSERT_EQUAL(thrust::get<3>(y), thrust::get<3>(abcd));

        
    //Empty in back
    y = tuple_cat(abcd, x);
    ASSERT_EQUAL(thrust::get<0>(y), thrust::get<0>(abcd));
    ASSERT_EQUAL(thrust::get<1>(y), thrust::get<1>(abcd));
    ASSERT_EQUAL(thrust::get<2>(y), thrust::get<2>(abcd));
    ASSERT_EQUAL(thrust::get<3>(y), thrust::get<3>(abcd));
    
    //Test concatenating up to maximum tuple size
    typedef thrust::tuple<int, int, int, int, int, int> iiiiii;
    iiiiii six_i = thrust::make_tuple(1,2,3,4,5,6);
    typedef thrust::tuple<A, B, C, D, int, int, int, int, int, int> maximum_tuple;
    maximum_tuple m = thrust::tuple_cat(abcd, six_i);
    ASSERT_EQUAL(thrust::get<0>(m), a);
    ASSERT_EQUAL(thrust::get<1>(m), b);
    ASSERT_EQUAL(thrust::get<2>(m), c);
    ASSERT_EQUAL(thrust::get<3>(m), d);
    ASSERT_EQUAL(thrust::get<4>(m), 1);
    ASSERT_EQUAL(thrust::get<5>(m), 2);
    ASSERT_EQUAL(thrust::get<6>(m), 3);
    ASSERT_EQUAL(thrust::get<7>(m), 4);
    ASSERT_EQUAL(thrust::get<8>(m), 5);
    ASSERT_EQUAL(thrust::get<9>(m), 6);

                 
    //Concat empties with maximum
    m = tuple_cat(x, m);
    ASSERT_EQUAL(thrust::get<0>(m), a);
    ASSERT_EQUAL(thrust::get<1>(m), b);
    ASSERT_EQUAL(thrust::get<2>(m), c);
    ASSERT_EQUAL(thrust::get<3>(m), d);
    ASSERT_EQUAL(thrust::get<4>(m), 1);
    ASSERT_EQUAL(thrust::get<5>(m), 2);
    ASSERT_EQUAL(thrust::get<6>(m), 3);
    ASSERT_EQUAL(thrust::get<7>(m), 4);
    ASSERT_EQUAL(thrust::get<8>(m), 5);
    ASSERT_EQUAL(thrust::get<9>(m), 6);
    m = tuple_cat(m, x);
    ASSERT_EQUAL(thrust::get<0>(m), a);
    ASSERT_EQUAL(thrust::get<1>(m), b);
    ASSERT_EQUAL(thrust::get<2>(m), c);
    ASSERT_EQUAL(thrust::get<3>(m), d);
    ASSERT_EQUAL(thrust::get<4>(m), 1);
    ASSERT_EQUAL(thrust::get<5>(m), 2);
    ASSERT_EQUAL(thrust::get<6>(m), 3);
    ASSERT_EQUAL(thrust::get<7>(m), 4);
    ASSERT_EQUAL(thrust::get<8>(m), 5);
    ASSERT_EQUAL(thrust::get<9>(m), 6);

    //Test tuple_cat with references
    thrust::tuple_cat(thrust::tie(a, b), thrust::tie(c, d)) =
        thrust::make_tuple(5, 1.61f, 3, 1.414);
    ASSERT_EQUAL(a, 5);
    ASSERT_EQUAL(b, 1.61f);
    ASSERT_EQUAL(c, 3);
    ASSERT_EQUAL(d, 1.414);

    //Test concatenating empty tuples
    abcd = thrust::tuple_cat(thrust::tuple<>(), ab, thrust::tuple<>(), thrust::tuple<>(), cd);
    ASSERT_EQUAL(thrust::get<0>(abcd), thrust::get<0>(ab));
    ASSERT_EQUAL(thrust::get<1>(abcd), thrust::get<1>(ab));
    ASSERT_EQUAL(thrust::get<2>(abcd), thrust::get<0>(cd));
    ASSERT_EQUAL(thrust::get<3>(abcd), thrust::get<1>(cd));

    //Test concatenating pairs
    six_i = thrust::tuple_cat(thrust::make_pair(3, 2),
                              thrust::make_pair(4, 5),
                              thrust::make_pair(6, 7));
    ASSERT_EQUAL(thrust::get<0>(six_i), 3);
    ASSERT_EQUAL(thrust::get<1>(six_i), 2);
    ASSERT_EQUAL(thrust::get<2>(six_i), 4);
    ASSERT_EQUAL(thrust::get<3>(six_i), 5);
    ASSERT_EQUAL(thrust::get<4>(six_i), 6);
    ASSERT_EQUAL(thrust::get<5>(six_i), 7);

    
}
DECLARE_UNITTEST(TestTupleCat);
