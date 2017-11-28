#pragma once

#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>
#include <tbb/parallel_sort.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/tick_count.h>
#include <tbb/tbb_thread.h>

#include <cassert>

// TBB bodies
template <typename T>
class NegateBody
{ 
    public:
    void operator()(T& x) const
    {
        x = -x;
    }
};

template <typename Vector>
class ForBody
{ 
    Vector &v;
    typedef typename Vector::value_type T;

    public: 
    ForBody(Vector& x) : v(x) {}    

    void operator()(const tbb::blocked_range<size_t>& r) const
    { 
        for(size_t i=r.begin(); i != r.end(); ++i)  
            v[i] = -v[i];
    }
};

template <typename Vector>
class ReduceBody
{ 
    Vector &v;
    typedef typename Vector::value_type T;

    public: 
    T sum;  
    void operator()(const tbb::blocked_range<size_t>& r )
    { 
        for(size_t i=r.begin(); i != r.end(); ++i)  
            sum += v[i];
    }
    
    ReduceBody(ReduceBody& x, tbb::split) : v(x.v), sum(0) {}
    void join(const ReduceBody& y ) { sum += y.sum; } 
    ReduceBody(Vector& x) : v(x), sum(0) {}    
};

template <typename Vector>
class ScanBody
{ 
    typedef typename Vector::value_type T;
    Vector& x; 
public: 
    T sum; 
    ScanBody(Vector& x) : sum(0), x(x) {} 
    T get_sum() const {return sum;} 
    template<typename Tag> 
    void operator()(const tbb::blocked_range<size_t>& r, Tag)
    {
        T temp = sum; 
        for(size_t i = r.begin(); i < r.end(); ++i)
        { 
            temp = temp + x[i]; 
            if(Tag::is_final_scan()) 
                x[i] = temp; 
        }        
        sum = temp; 
    }
    ScanBody(ScanBody& b, tbb::split) : x(b.x), sum(0) {} 
    void reverse_join(ScanBody& a) { sum = a.sum + sum;} 
    void assign(ScanBody& b) { sum = b.sum; } 
};

template <typename Vector>
typename Vector::value_type tbb_reduce(Vector& v)
{
    ReduceBody<Vector> body(v);

    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, v.size()), body);

    return body.sum;
}

template <typename Vector>
void tbb_transform(Vector& v)
{
    ForBody<Vector> body(v);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, v.size()), body);
}

template <typename Vector>
void tbb_scan(Vector& v)
{
    ScanBody<Vector> body(v);
    tbb::parallel_scan(tbb::blocked_range<size_t>(0, v.size()), body);
}

template <typename Vector>
void tbb_sort(Vector& v)
{
    tbb::parallel_sort(v.begin(), v.end());
}


void test_tbb(void)
{
    size_t n = 1 << 20;
    std::vector<int> A(n);
    std::vector<int> B(n);

    randomize(A);
    randomize(B);
    assert(std::accumulate(A.begin(), A.end(), 0) == tbb_reduce(A));
    
    randomize(A);
    randomize(B);
    std::transform(A.begin(), A.end(), A.begin(), thrust::negate<int>());
    tbb_transform(B);
    assert(A == B);
   
    randomize(A);
    randomize(B);
    std::partial_sum(A.begin(), A.end(), A.begin());
    tbb_scan(B);
    assert(A == B);

    randomize(A);
    randomize(B);
    std::sort(A.begin(), A.end());
    tbb_sort(B);
    assert(A == B);

    //printf("[Test: TBB algorithms OK]\n");
}

