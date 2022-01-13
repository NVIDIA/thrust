---
title: Modifying
parent: Transformations
grand_parent: Algorithms
nav_exclude: false
has_children: true
has_toc: false
---

# Modifying

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction&gt;</span>
<span>__host__ __device__ InputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__modifying.html#function-for-each">thrust::for&#95;each</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;UnaryFunction f);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Size,</span>
<span>&nbsp;&nbsp;typename UnaryFunction&gt;</span>
<span>__host__ __device__ InputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__modifying.html#function-for-each-n">thrust::for&#95;each&#95;n</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;Size n,</span>
<span>&nbsp;&nbsp;UnaryFunction f);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction&gt;</span>
<span>InputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__modifying.html#function-for-each">thrust::for&#95;each</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;UnaryFunction f);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Size,</span>
<span>&nbsp;&nbsp;typename UnaryFunction&gt;</span>
<span>InputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__modifying.html#function-for-each-n">thrust::for&#95;each&#95;n</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;Size n,</span>
<span>&nbsp;&nbsp;UnaryFunction f);</span>
</code>

## Functions

<h3 id="function-for-each">
Function <code>thrust::for&#95;each</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction&gt;</span>
<span>__host__ __device__ InputIterator </span><span><b>for_each</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;UnaryFunction f);</span></code>
<code>for&#95;each</code> applies the function object <code>f</code> to each element in the range <code>[first, last)</code>; <code>f's</code> return value, if any, is ignored. Unlike the C++ Standard Template Library function <code>std::for&#95;each</code>, this version offers no guarantee on order of execution. For this reason, this version of <code>for&#95;each</code> does not return a copy of the function object.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>for&#95;each</code> to print the elements of a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">thrust::device&#95;vector</a></code> using the <code>thrust::device</code> parallelization policy:



```cpp
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <cstdio>
...

struct printf_functor
{
  __host__ __device__
  void operator()(int x)
  {
    // note that using printf in a __device__ function requires
    // code compiled for a GPU with compute capability 2.0 or
    // higher (nvcc --arch=sm_20)
    printf("%d\n", x);
  }
};
...
thrust::device_vector<int> d_vec(3);
d_vec[0] = 0; d_vec[1] = 1; d_vec[2] = 2;

thrust::for_each(thrust::device, d_vec.begin(), d_vec.end(), printf_functor());

// 0 1 2 is printed to standard output in some unspecified order
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/InputIterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>UnaryFunction's</code><code>argument&#95;type</code>. 
* **`UnaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a>, and <code>UnaryFunction</code> does not apply any non-constant operation through its argument.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`f`** The function object to apply to the range <code>[first, last)</code>. 

**Returns**:
last

**See**:
* for_each_n 
* <a href="https://en.cppreference.com/w/cpp/algorithm/for_each">https://en.cppreference.com/w/cpp/algorithm/for_each</a>

<h3 id="function-for-each-n">
Function <code>thrust::for&#95;each&#95;n</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Size,</span>
<span>&nbsp;&nbsp;typename UnaryFunction&gt;</span>
<span>__host__ __device__ InputIterator </span><span><b>for_each_n</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;Size n,</span>
<span>&nbsp;&nbsp;UnaryFunction f);</span></code>
<code>for&#95;each&#95;n</code> applies the function object <code>f</code> to each element in the range <code>[first, first + n)</code>; <code>f's</code> return value, if any, is ignored. Unlike the C++ Standard Template Library function <code>std::for&#95;each</code>, this version offers no guarantee on order of execution.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>for&#95;each&#95;n</code> to print the elements of a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device&#95;vector</a></code> using the <code>thrust::device</code> parallelization policy.



```cpp
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <cstdio>

struct printf_functor
{
  __host__ __device__
  void operator()(int x)
  {
    // note that using printf in a __device__ function requires
    // code compiled for a GPU with compute capability 2.0 or
    // higher (nvcc --arch=sm_20)
    printf("%d\n", x);
  }
};
...
thrust::device_vector<int> d_vec(3);
d_vec[0] = 0; d_vec[1] = 1; d_vec[2] = 2;

thrust::for_each_n(thrust::device, d_vec.begin(), d_vec.size(), printf_functor());

// 0 1 2 is printed to standard output in some unspecified order
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/InputIterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>UnaryFunction's</code><code>argument&#95;type</code>. 
* **`Size`** is an integral type. 
* **`UnaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a>, and <code>UnaryFunction</code> does not apply any non-constant operation through its argument.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`n`** The size of the input sequence. 
* **`f`** The function object to apply to the range <code>[first, first + n)</code>. 

**Returns**:
<code>first + n</code>

**See**:
* for_each 
* <a href="https://en.cppreference.com/w/cpp/algorithm/for_each">https://en.cppreference.com/w/cpp/algorithm/for_each</a>

<h3 id="function-for-each">
Function <code>thrust::for&#95;each</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction&gt;</span>
<span>InputIterator </span><span><b>for_each</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;UnaryFunction f);</span></code>
<code>for&#95;each</code> applies the function object <code>f</code> to each element in the range <code>[first, last)</code>; <code>f's</code> return value, if any, is ignored. Unlike the C++ Standard Template Library function <code>std::for&#95;each</code>, this version offers no guarantee on order of execution. For this reason, this version of <code>for&#95;each</code> does not return a copy of the function object.


The following code snippet demonstrates how to use <code>for&#95;each</code> to print the elements of a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device&#95;vector</a></code>.



```cpp
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <stdio.h>

struct printf_functor
{
  __host__ __device__
  void operator()(int x)
  {
    // note that using printf in a __device__ function requires
    // code compiled for a GPU with compute capability 2.0 or
    // higher (nvcc --arch=sm_20)
    printf("%d\n", x);
  }
};
...
thrust::device_vector<int> d_vec(3);
d_vec[0] = 0; d_vec[1] = 1; d_vec[2] = 2;

thrust::for_each(d_vec.begin(), d_vec.end(), printf_functor());

// 0 1 2 is printed to standard output in some unspecified order
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/InputIterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>UnaryFunction's</code><code>argument&#95;type</code>. 
* **`UnaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a>, and <code>UnaryFunction</code> does not apply any non-constant operation through its argument.

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`f`** The function object to apply to the range <code>[first, last)</code>. 

**Returns**:
last

**See**:
* for_each_n 
* <a href="https://en.cppreference.com/w/cpp/algorithm/for_each">https://en.cppreference.com/w/cpp/algorithm/for_each</a>

<h3 id="function-for-each-n">
Function <code>thrust::for&#95;each&#95;n</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Size,</span>
<span>&nbsp;&nbsp;typename UnaryFunction&gt;</span>
<span>InputIterator </span><span><b>for_each_n</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;Size n,</span>
<span>&nbsp;&nbsp;UnaryFunction f);</span></code>
<code>for&#95;each&#95;n</code> applies the function object <code>f</code> to each element in the range <code>[first, first + n)</code>; <code>f's</code> return value, if any, is ignored. Unlike the C++ Standard Template Library function <code>std::for&#95;each</code>, this version offers no guarantee on order of execution.


The following code snippet demonstrates how to use <code>for&#95;each&#95;n</code> to print the elements of a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device&#95;vector</a></code>.



```cpp
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <stdio.h>

struct printf_functor
{
  __host__ __device__
  void operator()(int x)
  {
    // note that using printf in a __device__ function requires
    // code compiled for a GPU with compute capability 2.0 or
    // higher (nvcc --arch=sm_20)
    printf("%d\n", x);
  }
};
...
thrust::device_vector<int> d_vec(3);
d_vec[0] = 0; d_vec[1] = 1; d_vec[2] = 2;

thrust::for_each_n(d_vec.begin(), d_vec.size(), printf_functor());

// 0 1 2 is printed to standard output in some unspecified order
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/InputIterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>UnaryFunction's</code><code>argument&#95;type</code>. 
* **`Size`** is an integral type. 
* **`UnaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a>, and <code>UnaryFunction</code> does not apply any non-constant operation through its argument.

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`n`** The size of the input sequence. 
* **`f`** The function object to apply to the range <code>[first, first + n)</code>. 

**Returns**:
<code>first + n</code>

**See**:
* for_each 
* <a href="https://en.cppreference.com/w/cpp/algorithm/for_each">https://en.cppreference.com/w/cpp/algorithm/for_each</a>


