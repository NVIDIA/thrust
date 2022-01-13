---
title: Transformed Reductions
parent: Reductions
grand_parent: Algorithms
nav_exclude: false
has_children: true
has_toc: false
---

# Transformed Reductions

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputType&gt;</span>
<span>__host__ __device__ OutputType </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformed__reductions.html#function-inner-product">thrust::inner&#95;product</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputType init);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputType&gt;</span>
<span>OutputType </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformed__reductions.html#function-inner-product">thrust::inner&#95;product</a></b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputType init);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputType,</span>
<span>&nbsp;&nbsp;typename BinaryFunction1,</span>
<span>&nbsp;&nbsp;typename BinaryFunction2&gt;</span>
<span>__host__ __device__ OutputType </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformed__reductions.html#function-inner-product">thrust::inner&#95;product</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputType init,</span>
<span>&nbsp;&nbsp;BinaryFunction1 binary_op1,</span>
<span>&nbsp;&nbsp;BinaryFunction2 binary_op2);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputType,</span>
<span>&nbsp;&nbsp;typename BinaryFunction1,</span>
<span>&nbsp;&nbsp;typename BinaryFunction2&gt;</span>
<span>OutputType </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformed__reductions.html#function-inner-product">thrust::inner&#95;product</a></b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputType init,</span>
<span>&nbsp;&nbsp;BinaryFunction1 binary_op1,</span>
<span>&nbsp;&nbsp;BinaryFunction2 binary_op2);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction,</span>
<span>&nbsp;&nbsp;typename OutputType,</span>
<span>&nbsp;&nbsp;typename BinaryFunction&gt;</span>
<span>__host__ __device__ OutputType </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformed__reductions.html#function-transform-reduce">thrust::transform&#95;reduce</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;UnaryFunction unary_op,</span>
<span>&nbsp;&nbsp;OutputType init,</span>
<span>&nbsp;&nbsp;BinaryFunction binary_op);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction,</span>
<span>&nbsp;&nbsp;typename OutputType,</span>
<span>&nbsp;&nbsp;typename BinaryFunction&gt;</span>
<span>OutputType </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformed__reductions.html#function-transform-reduce">thrust::transform&#95;reduce</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;UnaryFunction unary_op,</span>
<span>&nbsp;&nbsp;OutputType init,</span>
<span>&nbsp;&nbsp;BinaryFunction binary_op);</span>
</code>

## Functions

<h3 id="function-inner-product">
Function <code>thrust::inner&#95;product</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputType&gt;</span>
<span>__host__ __device__ OutputType </span><span><b>inner_product</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputType init);</span></code>
<code>inner&#95;product</code> calculates an inner product of the ranges <code>[first1, last1)</code> and <code>[first2, first2 + (last1 - first1))</code>.

Specifically, this version of <code>inner&#95;product</code> computes the sum <code>init + (&#42;first1 &#42; &#42;first2) + (&#42;(first1+1) &#42; &#42;(first2+1)) + ... </code>

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code demonstrates how to use <code>inner&#95;product</code> to compute the dot product of two vectors using the <code>thrust::host</code> execution policy for parallelization.



```cpp
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
...
float vec1[3] = {1.0f, 2.0f, 5.0f};
float vec2[3] = {4.0f, 1.0f, 5.0f};

float result = thrust::inner_product(thrust::host, vec1, vec1 + 3, vec2, 0.0f);

// result == 31.0f
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, 
* **`OutputType`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and if <code>x</code> is an object of type <code>OutputType</code>, and <code>y</code> is an object of <code>InputIterator1's</code><code>value&#95;type</code>, and <code>z</code> is an object of <code>InputIterator2's</code><code>value&#95;type</code>, then <code>x + y &#42; z</code> is defined and is convertible to <code>OutputType</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the first sequence. 
* **`last1`** The end of the first sequence. 
* **`first2`** The beginning of the second sequence. 
* **`init`** Initial value of the result. 

**Returns**:
The inner product of sequences <code>[first1, last1)</code> and <code>[first2, last2)</code> plus <code>init</code>.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/inner_product">https://en.cppreference.com/w/cpp/algorithm/inner_product</a>

<h3 id="function-inner-product">
Function <code>thrust::inner&#95;product</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputType&gt;</span>
<span>OutputType </span><span><b>inner_product</b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputType init);</span></code>
<code>inner&#95;product</code> calculates an inner product of the ranges <code>[first1, last1)</code> and <code>[first2, first2 + (last1 - first1))</code>.

Specifically, this version of <code>inner&#95;product</code> computes the sum <code>init + (&#42;first1 &#42; &#42;first2) + (&#42;(first1+1) &#42; &#42;(first2+1)) + ... </code>

Unlike the C++ Standard Template Library function <code>std::inner&#95;product</code>, this version offers no guarantee on order of execution.


The following code demonstrates how to use <code>inner&#95;product</code> to compute the dot product of two vectors.



```cpp
#include <thrust/inner_product.h>
...
float vec1[3] = {1.0f, 2.0f, 5.0f};
float vec2[3] = {4.0f, 1.0f, 5.0f};

float result = thrust::inner_product(vec1, vec1 + 3, vec2, 0.0f);

// result == 31.0f
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, 
* **`OutputType`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and if <code>x</code> is an object of type <code>OutputType</code>, and <code>y</code> is an object of <code>InputIterator1's</code><code>value&#95;type</code>, and <code>z</code> is an object of <code>InputIterator2's</code><code>value&#95;type</code>, then <code>x + y &#42; z</code> is defined and is convertible to <code>OutputType</code>.

**Function Parameters**:
* **`first1`** The beginning of the first sequence. 
* **`last1`** The end of the first sequence. 
* **`first2`** The beginning of the second sequence. 
* **`init`** Initial value of the result. 

**Returns**:
The inner product of sequences <code>[first1, last1)</code> and <code>[first2, last2)</code> plus <code>init</code>.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/inner_product">https://en.cppreference.com/w/cpp/algorithm/inner_product</a>

<h3 id="function-inner-product">
Function <code>thrust::inner&#95;product</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputType,</span>
<span>&nbsp;&nbsp;typename BinaryFunction1,</span>
<span>&nbsp;&nbsp;typename BinaryFunction2&gt;</span>
<span>__host__ __device__ OutputType </span><span><b>inner_product</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputType init,</span>
<span>&nbsp;&nbsp;BinaryFunction1 binary_op1,</span>
<span>&nbsp;&nbsp;BinaryFunction2 binary_op2);</span></code>
<code>inner&#95;product</code> calculates an inner product of the ranges <code>[first1, last1)</code> and <code>[first2, first2 + (last1 - first1))</code>.

This version of <code>inner&#95;product</code> is identical to the first, except that is uses two user-supplied function objects instead of <code>operator+</code> and <code>operator&#42;</code>.

Specifically, this version of <code>inner&#95;product</code> computes the sum <code>binary&#95;op1( init, binary&#95;op2(&#42;first1, &#42;first2) ), ... </code>

The algorithm's execution is parallelized as determined by <code>exec</code>.



```cpp
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
...
float vec1[3] = {1.0f, 2.0f, 5.0f};
float vec2[3] = {4.0f, 1.0f, 5.0f};

float init = 0.0f;
thrust::plus<float>       binary_op1;
thrust::multiplies<float> binary_op2;

float result = thrust::inner_product(thrust::host, vec1, vec1 + 3, vec2, init, binary_op1, binary_op2);

// result == 31.0f
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator1's</code><code>value&#95;type</code> is convertible to <code>BinaryFunction2's</code><code>first&#95;argument&#95;type</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>BinaryFunction2's</code><code>second&#95;argument&#95;type</code>. 
* **`OutputType`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and <code>OutputType</code> is convertible to <code>BinaryFunction1's</code><code>first&#95;argument&#95;type</code>. 
* **`BinaryFunction1`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>, and <code>BinaryFunction1's</code><code>return&#95;type</code> is convertible to <code>OutputType</code>. 
* **`BinaryFunction2`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>, and <code>BinaryFunction2's</code><code>return&#95;type</code> is convertible to <code>BinaryFunction1's</code><code>second&#95;argument&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the first sequence. 
* **`last1`** The end of the first sequence. 
* **`first2`** The beginning of the second sequence. 
* **`init`** Initial value of the result. 
* **`binary_op1`** Generalized addition operation. 
* **`binary_op2`** Generalized multiplication operation. 

**Returns**:
The inner product of sequences <code>[first1, last1)</code> and <code>[first2, last2)</code>.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/inner_product">https://en.cppreference.com/w/cpp/algorithm/inner_product</a>

<h3 id="function-inner-product">
Function <code>thrust::inner&#95;product</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputType,</span>
<span>&nbsp;&nbsp;typename BinaryFunction1,</span>
<span>&nbsp;&nbsp;typename BinaryFunction2&gt;</span>
<span>OutputType </span><span><b>inner_product</b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputType init,</span>
<span>&nbsp;&nbsp;BinaryFunction1 binary_op1,</span>
<span>&nbsp;&nbsp;BinaryFunction2 binary_op2);</span></code>
<code>inner&#95;product</code> calculates an inner product of the ranges <code>[first1, last1)</code> and <code>[first2, first2 + (last1 - first1))</code>.

This version of <code>inner&#95;product</code> is identical to the first, except that is uses two user-supplied function objects instead of <code>operator+</code> and <code>operator&#42;</code>.

Specifically, this version of <code>inner&#95;product</code> computes the sum <code>binary&#95;op1( init, binary&#95;op2(&#42;first1, &#42;first2) ), ... </code>

Unlike the C++ Standard Template Library function <code>std::inner&#95;product</code>, this version offers no guarantee on order of execution.



```cpp
#include <thrust/inner_product.h>
...
float vec1[3] = {1.0f, 2.0f, 5.0f};
float vec2[3] = {4.0f, 1.0f, 5.0f};

float init = 0.0f;
thrust::plus<float>       binary_op1;
thrust::multiplies<float> binary_op2;

float result = thrust::inner_product(vec1, vec1 + 3, vec2, init, binary_op1, binary_op2);

// result == 31.0f
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator1's</code><code>value&#95;type</code> is convertible to <code>BinaryFunction2's</code><code>first&#95;argument&#95;type</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>BinaryFunction2's</code><code>second&#95;argument&#95;type</code>. 
* **`OutputType`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and <code>OutputType</code> is convertible to <code>BinaryFunction1's</code><code>first&#95;argument&#95;type</code>. 
* **`BinaryFunction1`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>, and <code>BinaryFunction1's</code><code>return&#95;type</code> is convertible to <code>OutputType</code>. 
* **`BinaryFunction2`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>, and <code>BinaryFunction2's</code><code>return&#95;type</code> is convertible to <code>BinaryFunction1's</code><code>second&#95;argument&#95;type</code>.

**Function Parameters**:
* **`first1`** The beginning of the first sequence. 
* **`last1`** The end of the first sequence. 
* **`first2`** The beginning of the second sequence. 
* **`init`** Initial value of the result. 
* **`binary_op1`** Generalized addition operation. 
* **`binary_op2`** Generalized multiplication operation. 

**Returns**:
The inner product of sequences <code>[first1, last1)</code> and <code>[first2, last2)</code>.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/inner_product">https://en.cppreference.com/w/cpp/algorithm/inner_product</a>

<h3 id="function-transform-reduce">
Function <code>thrust::transform&#95;reduce</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction,</span>
<span>&nbsp;&nbsp;typename OutputType,</span>
<span>&nbsp;&nbsp;typename BinaryFunction&gt;</span>
<span>__host__ __device__ OutputType </span><span><b>transform_reduce</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;UnaryFunction unary_op,</span>
<span>&nbsp;&nbsp;OutputType init,</span>
<span>&nbsp;&nbsp;BinaryFunction binary_op);</span></code>
<code>transform&#95;reduce</code> fuses the <code>transform</code> and <code>reduce</code> operations. <code>transform&#95;reduce</code> is equivalent to performing a transformation defined by <code>unary&#95;op</code> into a temporary sequence and then performing <code>reduce</code> on the transformed sequence. In most cases, fusing these two operations together is more efficient, since fewer memory reads and writes are required.

<code>transform&#95;reduce</code> performs a reduction on the transformation of the sequence <code>[first, last)</code> according to <code>unary&#95;op</code>. Specifically, <code>unary&#95;op</code> is applied to each element of the sequence and then the result is reduced to a single value with <code>binary&#95;op</code> using the initial value <code>init</code>. Note that the transformation <code>unary&#95;op</code> is not applied to the initial value <code>init</code>. The order of reduction is not specified, so <code>binary&#95;op</code> must be both commutative and associative.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>transform&#95;reduce</code> to compute the maximum value of the absolute value of the elements of a range using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

template<typename T>
struct absolute_value : public unary_function<T,T>
{
  __host__ __device__ T operator()(const T &x) const
  {
    return x < T(0) ? -x : x;
  }
};

...

int data[6] = {-1, 0, -2, -2, 1, -3};
int result = thrust::transform_reduce(thrust::host,
                                      data, data + 6,
                                      absolute_value<int>(),
                                      0,
                                      thrust::maximum<int>());
// result == 3
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>UnaryFunction's</code><code>argument&#95;type</code>. 
* **`UnaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a>, and <code>UnaryFunction's</code><code>result&#95;type</code> is convertible to <code>OutputType</code>. 
* **`OutputType`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and is convertible to <code>BinaryFunction's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>. 
* **`BinaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>, and <code>BinaryFunction's</code><code>result&#95;type</code> is convertible to <code>OutputType</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`unary_op`** The function to apply to each element of the input sequence. 
* **`init`** The result is initialized to this value. 
* **`binary_op`** The reduction operation. 

**Returns**:
The result of the transformed reduction.

**See**:
* <code>transform</code>
* <code>reduce</code>

<h3 id="function-transform-reduce">
Function <code>thrust::transform&#95;reduce</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction,</span>
<span>&nbsp;&nbsp;typename OutputType,</span>
<span>&nbsp;&nbsp;typename BinaryFunction&gt;</span>
<span>OutputType </span><span><b>transform_reduce</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;UnaryFunction unary_op,</span>
<span>&nbsp;&nbsp;OutputType init,</span>
<span>&nbsp;&nbsp;BinaryFunction binary_op);</span></code>
<code>transform&#95;reduce</code> fuses the <code>transform</code> and <code>reduce</code> operations. <code>transform&#95;reduce</code> is equivalent to performing a transformation defined by <code>unary&#95;op</code> into a temporary sequence and then performing <code>reduce</code> on the transformed sequence. In most cases, fusing these two operations together is more efficient, since fewer memory reads and writes are required.

<code>transform&#95;reduce</code> performs a reduction on the transformation of the sequence <code>[first, last)</code> according to <code>unary&#95;op</code>. Specifically, <code>unary&#95;op</code> is applied to each element of the sequence and then the result is reduced to a single value with <code>binary&#95;op</code> using the initial value <code>init</code>. Note that the transformation <code>unary&#95;op</code> is not applied to the initial value <code>init</code>. The order of reduction is not specified, so <code>binary&#95;op</code> must be both commutative and associative.


The following code snippet demonstrates how to use <code>transform&#95;reduce</code> to compute the maximum value of the absolute value of the elements of a range.



```cpp
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

template<typename T>
struct absolute_value : public unary_function<T,T>
{
  __host__ __device__ T operator()(const T &x) const
  {
    return x < T(0) ? -x : x;
  }
};

...

int data[6] = {-1, 0, -2, -2, 1, -3};
int result = thrust::transform_reduce(data, data + 6,
                                      absolute_value<int>(),
                                      0,
                                      thrust::maximum<int>());
// result == 3
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>UnaryFunction's</code><code>argument&#95;type</code>. 
* **`UnaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a>, and <code>UnaryFunction's</code><code>result&#95;type</code> is convertible to <code>OutputType</code>. 
* **`OutputType`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and is convertible to <code>BinaryFunction's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>. 
* **`BinaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>, and <code>BinaryFunction's</code><code>result&#95;type</code> is convertible to <code>OutputType</code>.

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`unary_op`** The function to apply to each element of the input sequence. 
* **`init`** The result is initialized to this value. 
* **`binary_op`** The reduction operation. 

**Returns**:
The result of the transformed reduction.

**See**:
* <code>transform</code>
* <code>reduce</code>


