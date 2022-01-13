---
title: thrust::zip_function
parent: Function Object Adaptors
grand_parent: Function Objects
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::zip_function`

<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__function.html">zip&#95;function</a></code> is a function object that allows the easy use of N-ary function objects with <code>zip&#95;iterators</code> without redefining them to take a <code>tuple</code> instead of N arguments.

This means that if a functor that takes 2 arguments which could be used with the <code>transform</code> function and <code>device&#95;iterators</code> can be extended to take 3 arguments and <code>zip&#95;iterators</code> without rewriting the functor in terms of <code>tuple</code>.

The <code>make&#95;zip&#95;function</code> convenience function is provided to avoid having to explicitely define the type of the functor when creating a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__function.html">zip&#95;function</a></code>, whic is especially helpful when using lambdas as the functor.



```cpp
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/zip_function.h>

struct SumTuple {
  float operator()(Tuple tup) {
    return std::get<0>(tup) + std::get<1>(tup) + std::get<2>(tup);
  }
};
struct SumArgs {
  float operator()(float a, float b, float c) {
    return a + b + c;
  }
};

int main() {
  thrust::device_vector<float> A(3);
  thrust::device_vector<float> B(3);
  thrust::device_vector<float> C(3);
  thrust::device_vector<float> D(3);
  A[0] = 0.f; A[1] = 1.f; A[2] = 2.f;
  B[0] = 1.f; B[1] = 2.f; B[2] = 3.f;
  C[0] = 2.f; C[1] = 3.f; C[2] = 4.f;

  // The following four invocations of transform are equivalent
  // Transform with 3-tuple
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin(), C.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(A.end(), B.end(), C.end())),
                    D.begin(),
                    SumTuple{});

  // Transform with 3 parameters
  thrust::zip_function<SumArgs> adapted{};
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin(), C.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(A.end(), B.end(), C.end())),
                    D.begin(),
                    adapted);

  // Transform with 3 parameters with convenience function
  thrust::zip_function<SumArgs> adapted{};
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin(), C.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(A.end(), B.end(), C.end())),
                    D.begin(),
                    thrust::make_zip_function(SumArgs{}));

  // Transform with 3 parameters with convenience function and lambda
  thrust::zip_function<SumArgs> adapted{};
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin(), C.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(A.end(), B.end(), C.end())),
                    D.begin(),
                    thrust::make_zip_function([] (float a, float b, float c) {
                                                return a + b + c;
                                              }));
  return 0;
}
```

**See**:
* make_zip_function 
* <a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__iterator.html">zip_iterator</a>

<code class="doxybook">
<span>#include <thrust/zip_function.h></span><br>
<span>template &lt;typename Function&gt;</span>
<span>class thrust::zip&#95;function {</span>
<span>public:</span><span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__function.html#function-zip-function">zip&#95;function</a></b>(Function func);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename Tuple&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ decltype(auto) </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__function.html#function-operator()">operator()</a></b>(Tuple && args) const;</span>
<span>};</span>
</code>

## Member Functions

<h3 id="function-zip-function">
Function <code>thrust::zip&#95;function::zip&#95;function</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>zip_function</b>(Function func);</span></code>
<h3 id="function-operator()">
Function <code>thrust::zip&#95;function::operator()</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Tuple&gt;</span>
<span>__host__ __device__ decltype(auto) </span><span><b>operator()</b>(Tuple && args) const;</span></code>

