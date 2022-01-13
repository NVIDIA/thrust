---
title: Transformations
parent: Algorithms
grand_parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Transformations

## Groups

* **[Filling]({{ site.baseurl }}/api/groups/group__filling.html)**
* **[Modifying]({{ site.baseurl }}/api/groups/group__modifying.html)**
* **[Replacing]({{ site.baseurl }}/api/groups/group__replacing.html)**

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-adjacent-difference">thrust::adjacent&#95;difference</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename BinaryFunction&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-adjacent-difference">thrust::adjacent&#95;difference</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;BinaryFunction binary_op);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-adjacent-difference">thrust::adjacent&#95;difference</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename BinaryFunction&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-adjacent-difference">thrust::adjacent&#95;difference</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;BinaryFunction binary_op);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Generator&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-generate">thrust::generate</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Generator gen);</span>
<br>
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Generator&gt;</span>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-generate">thrust::generate</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Generator gen);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Size,</span>
<span>&nbsp;&nbsp;typename Generator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-generate-n">thrust::generate&#95;n</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;OutputIterator first,</span>
<span>&nbsp;&nbsp;Size n,</span>
<span>&nbsp;&nbsp;Generator gen);</span>
<br>
<span>template &lt;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Size,</span>
<span>&nbsp;&nbsp;typename Generator&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-generate-n">thrust::generate&#95;n</a></b>(OutputIterator first,</span>
<span>&nbsp;&nbsp;Size n,</span>
<span>&nbsp;&nbsp;Generator gen);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-sequence">thrust::sequence</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span>
<br>
<span>template &lt;typename ForwardIterator&gt;</span>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-sequence">thrust::sequence</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-sequence">thrust::sequence</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;T init);</span>
<br>
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-sequence">thrust::sequence</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;T init);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-sequence">thrust::sequence</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;T init,</span>
<span>&nbsp;&nbsp;T step);</span>
<br>
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-sequence">thrust::sequence</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;T init,</span>
<span>&nbsp;&nbsp;T step);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename UnaryOperation&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-tabulate">thrust::tabulate</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;UnaryOperation unary_op);</span>
<br>
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename UnaryOperation&gt;</span>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-tabulate">thrust::tabulate</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;UnaryOperation unary_op);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-transform">thrust::transform</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;UnaryFunction op);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-transform">thrust::transform</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;UnaryFunction op);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename BinaryFunction&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-transform">thrust::transform</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;BinaryFunction op);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename BinaryFunction&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-transform">thrust::transform</a></b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;BinaryFunction op);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-transform-if">thrust::transform&#95;if</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;ForwardIterator result,</span>
<span>&nbsp;&nbsp;UnaryFunction op,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-transform-if">thrust::transform&#95;if</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;ForwardIterator result,</span>
<span>&nbsp;&nbsp;UnaryFunction op,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-transform-if">thrust::transform&#95;if</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;ForwardIterator result,</span>
<span>&nbsp;&nbsp;UnaryFunction op,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-transform-if">thrust::transform&#95;if</a></b>(InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;ForwardIterator result,</span>
<span>&nbsp;&nbsp;UnaryFunction op,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename BinaryFunction,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-transform-if">thrust::transform&#95;if</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator3 stencil,</span>
<span>&nbsp;&nbsp;ForwardIterator result,</span>
<span>&nbsp;&nbsp;BinaryFunction binary_op,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename BinaryFunction,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformations.html#function-transform-if">thrust::transform&#95;if</a></b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator3 stencil,</span>
<span>&nbsp;&nbsp;ForwardIterator result,</span>
<span>&nbsp;&nbsp;BinaryFunction binary_op,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
</code>

## Functions

<h3 id="function-adjacent-difference">
Function <code>thrust::adjacent&#95;difference</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>adjacent_difference</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>adjacent&#95;difference</code> calculates the differences of adjacent elements in the range <code>[first, last)</code>. That is, <code>&#42;first</code> is assigned to <code>&#42;result</code>, and, for each iterator <code>i</code> in the range <code>[first + 1, last)</code>, the difference of <code>&#42;i</code> and <code>&#42;(i - 1)</code> is assigned to <code>&#42;(result + (i - first))</code>.

This version of <code>adjacent&#95;difference</code> uses <code>operator-</code> to calculate differences.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>adjacent&#95;difference</code> to compute the difference between adjacent elements of a range using the <code>thrust::device</code> execution policy:



```cpp
#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
...
int h_data[8] = {1, 2, 1, 2, 1, 2, 1, 2};
thrust::device_vector<int> d_data(h_data, h_data + 8);
thrust::device_vector<int> d_result(8);

thrust::adjacent_difference(thrust::device, d_data.begin(), d_data.end(), d_result.begin());

// d_result is now [1, 1, -1, 1, -1, 1, -1, 1]
```

**Remark**:
Note that <code>result</code> is permitted to be the same iterator as <code>first</code>. This is useful for computing differences "in place".

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>x</code> and <code>y</code> are objects of <code>InputIterator's</code><code>value&#95;type</code>, then <code>x</code> - <code>is</code> defined, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>, and the return type of <code>x - y</code> is convertible to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the input range. 
* **`last`** The end of the input range. 
* **`result`** The beginning of the output range. 

**Returns**:
The iterator <code>result + (last - first)</code>

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/adjacent_difference">https://en.cppreference.com/w/cpp/algorithm/adjacent_difference</a>
* inclusive_scan 

<h3 id="function-adjacent-difference">
Function <code>thrust::adjacent&#95;difference</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename BinaryFunction&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>adjacent_difference</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;BinaryFunction binary_op);</span></code>
<code>adjacent&#95;difference</code> calculates the differences of adjacent elements in the range <code>[first, last)</code>. That is, <code>&#42;first</code> is assigned to <code>&#42;result</code>, and, for each iterator <code>i</code> in the range <code>[first + 1, last)</code>, <code>binary&#95;op(&#42;i, &#42;(i - 1))</code> is assigned to <code>&#42;(result + (i - first))</code>.

This version of <code>adjacent&#95;difference</code> uses the binary function <code>binary&#95;op</code> to calculate differences.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>adjacent&#95;difference</code> to compute the sum between adjacent elements of a range using the <code>thrust::device</code> execution policy:



```cpp
#include <thrust/adjacent_difference.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
...
int h_data[8] = {1, 2, 1, 2, 1, 2, 1, 2};
thrust::device_vector<int> d_data(h_data, h_data + 8);
thrust::device_vector<int> d_result(8);

thrust::adjacent_difference(thrust::device, d_data.begin(), d_data.end(), d_result.begin(), thrust::plus<int>());

// d_result is now [1, 3, 3, 3, 3, 3, 3, 3]
```

**Remark**:
Note that <code>result</code> is permitted to be the same iterator as <code>first</code>. This is useful for computing differences "in place".

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>BinaryFunction's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`BinaryFunction's`** <code>result&#95;type</code> is convertible to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the input range. 
* **`last`** The end of the input range. 
* **`result`** The beginning of the output range. 
* **`binary_op`** The binary function used to compute differences. 

**Returns**:
The iterator <code>result + (last - first)</code>

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/adjacent_difference">https://en.cppreference.com/w/cpp/algorithm/adjacent_difference</a>
* inclusive_scan 

<h3 id="function-adjacent-difference">
Function <code>thrust::adjacent&#95;difference</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b>adjacent_difference</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>adjacent&#95;difference</code> calculates the differences of adjacent elements in the range <code>[first, last)</code>. That is, <code>&#42;first</code> is assigned to <code>&#42;result</code>, and, for each iterator <code>i</code> in the range <code>[first + 1, last)</code>, the difference of <code>&#42;i</code> and <code>&#42;(i - 1)</code> is assigned to <code>&#42;(result + (i - first))</code>.

This version of <code>adjacent&#95;difference</code> uses <code>operator-</code> to calculate differences.


The following code snippet demonstrates how to use <code>adjacent&#95;difference</code> to compute the difference between adjacent elements of a range.



```cpp
#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
...
int h_data[8] = {1, 2, 1, 2, 1, 2, 1, 2};
thrust::device_vector<int> d_data(h_data, h_data + 8);
thrust::device_vector<int> d_result(8);

thrust::adjacent_difference(d_data.begin(), d_data.end(), d_result.begin());

// d_result is now [1, 1, -1, 1, -1, 1, -1, 1]
```

**Remark**:
Note that <code>result</code> is permitted to be the same iterator as <code>first</code>. This is useful for computing differences "in place".

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>x</code> and <code>y</code> are objects of <code>InputIterator's</code><code>value&#95;type</code>, then <code>x</code> - <code>is</code> defined, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>, and the return type of <code>x - y</code> is convertible to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`first`** The beginning of the input range. 
* **`last`** The end of the input range. 
* **`result`** The beginning of the output range. 

**Returns**:
The iterator <code>result + (last - first)</code>

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/adjacent_difference">https://en.cppreference.com/w/cpp/algorithm/adjacent_difference</a>
* inclusive_scan 

<h3 id="function-adjacent-difference">
Function <code>thrust::adjacent&#95;difference</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename BinaryFunction&gt;</span>
<span>OutputIterator </span><span><b>adjacent_difference</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;BinaryFunction binary_op);</span></code>
<code>adjacent&#95;difference</code> calculates the differences of adjacent elements in the range <code>[first, last)</code>. That is, <code>&#42;first</code> is assigned to <code>&#42;result</code>, and, for each iterator <code>i</code> in the range <code>[first + 1, last)</code>, <code>binary&#95;op(&#42;i, &#42;(i - 1))</code> is assigned to <code>&#42;(result + (i - first))</code>.

This version of <code>adjacent&#95;difference</code> uses the binary function <code>binary&#95;op</code> to calculate differences.


The following code snippet demonstrates how to use <code>adjacent&#95;difference</code> to compute the sum between adjacent elements of a range.



```cpp
#include <thrust/adjacent_difference.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
...
int h_data[8] = {1, 2, 1, 2, 1, 2, 1, 2};
thrust::device_vector<int> d_data(h_data, h_data + 8);
thrust::device_vector<int> d_result(8);

thrust::adjacent_difference(d_data.begin(), d_data.end(), d_result.begin(), thrust::plus<int>());

// d_result is now [1, 3, 3, 3, 3, 3, 3, 3]
```

**Remark**:
Note that <code>result</code> is permitted to be the same iterator as <code>first</code>. This is useful for computing differences "in place".

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>BinaryFunction's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`BinaryFunction's`** <code>result&#95;type</code> is convertible to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>.

**Function Parameters**:
* **`first`** The beginning of the input range. 
* **`last`** The end of the input range. 
* **`result`** The beginning of the output range. 
* **`binary_op`** The binary function used to compute differences. 

**Returns**:
The iterator <code>result + (last - first)</code>

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/adjacent_difference">https://en.cppreference.com/w/cpp/algorithm/adjacent_difference</a>
* inclusive_scan 

<h3 id="function-generate">
Function <code>thrust::generate</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Generator&gt;</span>
<span>__host__ __device__ void </span><span><b>generate</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Generator gen);</span></code>
<code>generate</code> assigns the result of invoking <code>gen</code>, a function object that takes no arguments, to each element in the range <code>[first,last)</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to fill a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host&#95;vector</a></code> with random numbers, using the standard C library function <code>rand</code> using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <cstdlib>
...
thrust::host_vector<int> v(10);
srand(13);
thrust::generate(thrust::host, v.begin(), v.end(), rand);

// the elements of v are now pseudo-random numbers
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable. 
* **`Generator`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional">Generator</a>, and <code>Generator's</code><code>result&#95;type</code> is convertible to <code>ForwardIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The first element in the range of interest. 
* **`last`** The last element in the range of interest. 
* **`gen`** A function argument, taking no parameters, used to generate values to assign to elements in the range <code>[first,last)</code>.

**See**:
* generate_n 
* <a href="https://en.cppreference.com/w/cpp/algorithm/generate">https://en.cppreference.com/w/cpp/algorithm/generate</a>

<h3 id="function-generate">
Function <code>thrust::generate</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Generator&gt;</span>
<span>void </span><span><b>generate</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Generator gen);</span></code>
<code>generate</code> assigns the result of invoking <code>gen</code>, a function object that takes no arguments, to each element in the range <code>[first,last)</code>.


The following code snippet demonstrates how to fill a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host&#95;vector</a></code> with random numbers, using the standard C library function <code>rand</code>.



```cpp
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <cstdlib>
...
thrust::host_vector<int> v(10);
srand(13);
thrust::generate(v.begin(), v.end(), rand);

// the elements of v are now pseudo-random numbers
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable. 
* **`Generator`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional">Generator</a>, and <code>Generator's</code><code>result&#95;type</code> is convertible to <code>ForwardIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The first element in the range of interest. 
* **`last`** The last element in the range of interest. 
* **`gen`** A function argument, taking no parameters, used to generate values to assign to elements in the range <code>[first,last)</code>.

**See**:
* generate_n 
* <a href="https://en.cppreference.com/w/cpp/algorithm/generate">https://en.cppreference.com/w/cpp/algorithm/generate</a>

<h3 id="function-generate-n">
Function <code>thrust::generate&#95;n</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Size,</span>
<span>&nbsp;&nbsp;typename Generator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>generate_n</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;OutputIterator first,</span>
<span>&nbsp;&nbsp;Size n,</span>
<span>&nbsp;&nbsp;Generator gen);</span></code>
<code>generate&#95;n</code> assigns the result of invoking <code>gen</code>, a function object that takes no arguments, to each element in the range <code>[first,first + n)</code>. The return value is <code>first + n</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to fill a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host&#95;vector</a></code> with random numbers, using the standard C library function <code>rand</code> using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <cstdlib>
...
thrust::host_vector<int> v(10);
srand(13);
thrust::generate_n(thrust::host, v.begin(), 10, rand);

// the elements of v are now pseudo-random numbers
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`Size`** is an integral type (either signed or unsigned). 
* **`Generator`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional">Generator</a>, and <code>Generator's</code><code>result&#95;type</code> is convertible to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The first element in the range of interest. 
* **`n`** The size of the range of interest. 
* **`gen`** A function argument, taking no parameters, used to generate values to assign to elements in the range <code>[first,first + n)</code>.

**See**:
* generate 
* <a href="https://en.cppreference.com/w/cpp/algorithm/generate">https://en.cppreference.com/w/cpp/algorithm/generate</a>

<h3 id="function-generate-n">
Function <code>thrust::generate&#95;n</code>
</h3>

<code class="doxybook">
<span>template &lt;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Size,</span>
<span>&nbsp;&nbsp;typename Generator&gt;</span>
<span>OutputIterator </span><span><b>generate_n</b>(OutputIterator first,</span>
<span>&nbsp;&nbsp;Size n,</span>
<span>&nbsp;&nbsp;Generator gen);</span></code>
<code>generate&#95;n</code> assigns the result of invoking <code>gen</code>, a function object that takes no arguments, to each element in the range <code>[first,first + n)</code>. The return value is <code>first + n</code>.


The following code snippet demonstrates how to fill a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host&#95;vector</a></code> with random numbers, using the standard C library function <code>rand</code>.



```cpp
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <stdlib.h>
...
thrust::host_vector<int> v(10);
srand(13);
thrust::generate_n(v.begin(), 10, rand);

// the elements of v are now pseudo-random numbers
```

**Template Parameters**:
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`Size`** is an integral type (either signed or unsigned). 
* **`Generator`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional">Generator</a>, and <code>Generator's</code><code>result&#95;type</code> is convertible to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>.

**Function Parameters**:
* **`first`** The first element in the range of interest. 
* **`n`** The size of the range of interest. 
* **`gen`** A function argument, taking no parameters, used to generate values to assign to elements in the range <code>[first,first + n)</code>.

**See**:
* generate 
* <a href="https://en.cppreference.com/w/cpp/algorithm/generate">https://en.cppreference.com/w/cpp/algorithm/generate</a>

<h3 id="function-sequence">
Function <code>thrust::sequence</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator&gt;</span>
<span>__host__ __device__ void </span><span><b>sequence</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span></code>
<code>sequence</code> fills the range <code>[first, last)</code> with a sequence of numbers.

For each iterator <code>i</code> in the range <code>[first, last)</code>, this version of <code>sequence</code> performs the assignment <code>&#42;i = (i - first)</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>sequence</code> to fill a range with a sequence of numbers using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
...
const int N = 10;
int A[N];
thrust::sequence(thrust::host, A, A + 10);
// A is now {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
```

**Note**:
Unlike the similar C++ STL function <code>std::iota</code>, <code>sequence</code> offers no guarantee on order of execution.

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable, and if <code>x</code> and <code>y</code> are objects of <code>ForwardIterator's</code><code>value&#95;type</code>, then <code>x + y</code> is defined, and if <code>T</code> is <code>ForwardIterator's</code><code>value&#95;type</code>, then <code>T(0)</code> is defined.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/iota">https://en.cppreference.com/w/cpp/algorithm/iota</a>

<h3 id="function-sequence">
Function <code>thrust::sequence</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator&gt;</span>
<span>void </span><span><b>sequence</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span></code>
<code>sequence</code> fills the range <code>[first, last)</code> with a sequence of numbers.

For each iterator <code>i</code> in the range <code>[first, last)</code>, this version of <code>sequence</code> performs the assignment <code>&#42;i = (i - first)</code>.


The following code snippet demonstrates how to use <code>sequence</code> to fill a range with a sequence of numbers.



```cpp
#include <thrust/sequence.h>
...
const int N = 10;
int A[N];
thrust::sequence(A, A + 10);
// A is now {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
```

**Note**:
Unlike the similar C++ STL function <code>std::iota</code>, <code>sequence</code> offers no guarantee on order of execution.

**Template Parameters**:
**`ForwardIterator`**: is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable, and if <code>x</code> and <code>y</code> are objects of <code>ForwardIterator's</code><code>value&#95;type</code>, then <code>x + y</code> is defined, and if <code>T</code> is <code>ForwardIterator's</code><code>value&#95;type</code>, then <code>T(0)</code> is defined.

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/iota">https://en.cppreference.com/w/cpp/algorithm/iota</a>

<h3 id="function-sequence">
Function <code>thrust::sequence</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ void </span><span><b>sequence</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;T init);</span></code>
<code>sequence</code> fills the range <code>[first, last)</code> with a sequence of numbers.

For each iterator <code>i</code> in the range <code>[first, last)</code>, this version of <code>sequence</code> performs the assignment <code>&#42;i = init + (i - first)</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>sequence</code> to fill a range with a sequence of numbers starting from the value 1 using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
...
const int N = 10;
int A[N];
thrust::sequence(thrust::host, A, A + 10, 1);
// A is now {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
```

**Note**:
Unlike the similar C++ STL function <code>std::iota</code>, <code>sequence</code> offers no guarantee on order of execution.

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable, and if <code>x</code> and <code>y</code> are objects of <code>ForwardIterator's</code><code>value&#95;type</code>, then <code>x + y</code> is defined, and if <code>T</code> is <code>ForwardIterator's</code><code>value&#95;type</code>, then <code>T(0)</code> is defined. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and <code>T</code> is convertible to <code>ForwardIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`init`** The first value of the sequence of numbers.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/iota">https://en.cppreference.com/w/cpp/algorithm/iota</a>

<h3 id="function-sequence">
Function <code>thrust::sequence</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>void </span><span><b>sequence</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;T init);</span></code>
<code>sequence</code> fills the range <code>[first, last)</code> with a sequence of numbers.

For each iterator <code>i</code> in the range <code>[first, last)</code>, this version of <code>sequence</code> performs the assignment <code>&#42;i = init + (i - first)</code>.


The following code snippet demonstrates how to use <code>sequence</code> to fill a range with a sequence of numbers starting from the value 1.



```cpp
#include <thrust/sequence.h>
...
const int N = 10;
int A[N];
thrust::sequence(A, A + 10, 1);
// A is now {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
```

**Note**:
Unlike the similar C++ STL function <code>std::iota</code>, <code>sequence</code> offers no guarantee on order of execution.

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable, and if <code>x</code> and <code>y</code> are objects of <code>ForwardIterator's</code><code>value&#95;type</code>, then <code>x + y</code> is defined, and if <code>T</code> is <code>ForwardIterator's</code><code>value&#95;type</code>, then <code>T(0)</code> is defined. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and <code>T</code> is convertible to <code>ForwardIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`init`** The first value of the sequence of numbers.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/iota">https://en.cppreference.com/w/cpp/algorithm/iota</a>

<h3 id="function-sequence">
Function <code>thrust::sequence</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ void </span><span><b>sequence</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;T init,</span>
<span>&nbsp;&nbsp;T step);</span></code>
<code>sequence</code> fills the range <code>[first, last)</code> with a sequence of numbers.

For each iterator <code>i</code> in the range <code>[first, last)</code>, this version of <code>sequence</code> performs the assignment <code>&#42;i = init + step &#42; (i - first)</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>sequence</code> to fill a range with a sequence of numbers starting from the value 1 with a step size of 3 using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
...
const int N = 10;
int A[N];
thrust::sequence(thrust::host, A, A + 10, 1, 3);
// A is now {1, 4, 7, 10, 13, 16, 19, 22, 25, 28}
```

**Note**:
Unlike the similar C++ STL function <code>std::iota</code>, <code>sequence</code> offers no guarantee on order of execution.

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable, and if <code>x</code> and <code>y</code> are objects of <code>ForwardIterator's</code><code>value&#95;type</code>, then <code>x + y</code> is defined, and if <code>T</code> is <code>ForwardIterator's</code><code>value&#95;type</code>, then <code>T(0)</code> is defined. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and <code>T</code> is convertible to <code>ForwardIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`init`** The first value of the sequence of numbers 
* **`step`** The difference between consecutive elements.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/iota">https://en.cppreference.com/w/cpp/algorithm/iota</a>

<h3 id="function-sequence">
Function <code>thrust::sequence</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>void </span><span><b>sequence</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;T init,</span>
<span>&nbsp;&nbsp;T step);</span></code>
<code>sequence</code> fills the range <code>[first, last)</code> with a sequence of numbers.

For each iterator <code>i</code> in the range <code>[first, last)</code>, this version of <code>sequence</code> performs the assignment <code>&#42;i = init + step &#42; (i - first)</code>.


The following code snippet demonstrates how to use <code>sequence</code> to fill a range with a sequence of numbers starting from the value 1 with a step size of 3.



```cpp
#include <thrust/sequence.h>
...
const int N = 10;
int A[N];
thrust::sequence(A, A + 10, 1, 3);
// A is now {1, 4, 7, 10, 13, 16, 19, 22, 25, 28}
```

**Note**:
Unlike the similar C++ STL function <code>std::iota</code>, <code>sequence</code> offers no guarantee on order of execution.

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable, and if <code>x</code> and <code>y</code> are objects of <code>ForwardIterator's</code><code>value&#95;type</code>, then <code>x + y</code> is defined, and if <code>T</code> is <code>ForwardIterator's</code><code>value&#95;type</code>, then <code>T(0)</code> is defined. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and <code>T</code> is convertible to <code>ForwardIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`init`** The first value of the sequence of numbers 
* **`step`** The difference between consecutive elements.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/iota">https://en.cppreference.com/w/cpp/algorithm/iota</a>

<h3 id="function-tabulate">
Function <code>thrust::tabulate</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename UnaryOperation&gt;</span>
<span>__host__ __device__ void </span><span><b>tabulate</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;UnaryOperation unary_op);</span></code>
<code>tabulate</code> fills the range <code>[first, last)</code> with the value of a function applied to each element's index.

For each iterator <code>i</code> in the range <code>[first, last)</code>, <code>tabulate</code> performs the assignment <code>&#42;i = unary&#95;op(i - first)</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>tabulate</code> to generate the first <code>n</code> non-positive integers using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/tabulate.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...
const int N = 10;
int A[N];
thrust::tabulate(thrust::host, A, A + 10, thrust::negate<int>());
// A is now {0, -1, -2, -3, -4, -5, -6, -7, -8, -9}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable, and if <code>x</code> and <code>y</code> are objects of <code>ForwardIterator's</code><code>value&#95;type</code>, then <code>x + y</code> is defined, and if <code>T</code> is <code>ForwardIterator's</code><code>value&#95;type</code>, then <code>T(0)</code> is defined. 
* **`UnaryOperation`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a> and <code>UnaryFunction's</code><code>result&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the range. 
* **`last`** The end of the range. 
* **`unary_op`** The unary operation to apply.

**See**:
* thrust::fill 
* thrust::generate 
* thrust::sequence 

<h3 id="function-tabulate">
Function <code>thrust::tabulate</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename UnaryOperation&gt;</span>
<span>void </span><span><b>tabulate</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;UnaryOperation unary_op);</span></code>
<code>tabulate</code> fills the range <code>[first, last)</code> with the value of a function applied to each element's index.

For each iterator <code>i</code> in the range <code>[first, last)</code>, <code>tabulate</code> performs the assignment <code>&#42;i = unary&#95;op(i - first)</code>.


The following code snippet demonstrates how to use <code>tabulate</code> to generate the first <code>n</code> non-positive integers:



```cpp
#include <thrust/tabulate.h>
#include <thrust/functional.h>
...
const int N = 10;
int A[N];
thrust::tabulate(A, A + 10, thrust::negate<int>());
// A is now {0, -1, -2, -3, -4, -5, -6, -7, -8, -9}
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable, and if <code>x</code> and <code>y</code> are objects of <code>ForwardIterator's</code><code>value&#95;type</code>, then <code>x + y</code> is defined, and if <code>T</code> is <code>ForwardIterator's</code><code>value&#95;type</code>, then <code>T(0)</code> is defined. 
* **`UnaryOperation`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a> and <code>UnaryFunction's</code><code>result&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The beginning of the range. 
* **`last`** The end of the range. 
* **`unary_op`** The unary operation to apply.

**See**:
* thrust::fill 
* thrust::generate 
* thrust::sequence 

<h3 id="function-transform">
Function <code>thrust::transform</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>transform</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;UnaryFunction op);</span></code>
This version of <code>transform</code> applies a unary function to each element of an input sequence and stores the result in the corresponding position in an output sequence. Specifically, for each iterator <code>i</code> in the range [<code>first</code>, <code>last</code>) the operation <code>op(&#42;i)</code> is performed and the result is assigned to <code>&#42;o</code>, where <code>o</code> is the corresponding output iterator in the range [<code>result</code>, <code>result</code> + (<code>last</code> - <code>first</code>) ). The input and output sequences may coincide, resulting in an in-place transformation.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>transform</code> to negate a range in-place using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...

int data[10] = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};

thrust::negate<int> op;

thrust::transform(thrust::host, data, data + 10, data, op); // in-place transformation

// data is now {5, 0, -2, 3, -2, -4, 0, 1, -2, -8};
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>UnaryFunction's</code><code>argument&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`UnaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a> and <code>UnaryFunction's</code><code>result&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the input sequence. 
* **`last`** The end of the input sequence. 
* **`result`** The beginning of the output sequence. 
* **`op`** The transformation operation. 

**Preconditions**:
<code>first</code> may equal <code>result</code>, but the range <code>[first, last)</code> shall not overlap the range <code>[result, result + (last - first))</code> otherwise.

**Returns**:
The end of the output sequence.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/transform">https://en.cppreference.com/w/cpp/algorithm/transform</a>

<h3 id="function-transform">
Function <code>thrust::transform</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction&gt;</span>
<span>OutputIterator </span><span><b>transform</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;UnaryFunction op);</span></code>
This version of <code>transform</code> applies a unary function to each element of an input sequence and stores the result in the corresponding position in an output sequence. Specifically, for each iterator <code>i</code> in the range [<code>first</code>, <code>last</code>) the operation <code>op(&#42;i)</code> is performed and the result is assigned to <code>&#42;o</code>, where <code>o</code> is the corresponding output iterator in the range [<code>result</code>, <code>result</code> + (<code>last</code> - <code>first</code>) ). The input and output sequences may coincide, resulting in an in-place transformation.


The following code snippet demonstrates how to use <code>transform</code>



```cpp
#include <thrust/transform.h>
#include <thrust/functional.h>

int data[10] = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};

thrust::negate<int> op;

thrust::transform(data, data + 10, data, op); // in-place transformation

// data is now {5, 0, -2, 3, -2, -4, 0, 1, -2, -8};
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>UnaryFunction's</code><code>argument&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`UnaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a> and <code>UnaryFunction's</code><code>result&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The beginning of the input sequence. 
* **`last`** The end of the input sequence. 
* **`result`** The beginning of the output sequence. 
* **`op`** The tranformation operation. 

**Preconditions**:
<code>first</code> may equal <code>result</code>, but the range <code>[first, last)</code> shall not overlap the range <code>[result, result + (last - first))</code> otherwise.

**Returns**:
The end of the output sequence.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/transform">https://en.cppreference.com/w/cpp/algorithm/transform</a>

<h3 id="function-transform">
Function <code>thrust::transform</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename BinaryFunction&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>transform</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;BinaryFunction op);</span></code>
This version of <code>transform</code> applies a binary function to each pair of elements from two input sequences and stores the result in the corresponding position in an output sequence. Specifically, for each iterator <code>i</code> in the range [<code>first1</code>, <code>last1</code>) and <code>j = first + (i - first1)</code> in the range [<code>first2</code>, <code>last2</code>) the operation <code>op(&#42;i,&#42;j)</code> is performed and the result is assigned to <code>&#42;o</code>, where <code>o</code> is the corresponding output iterator in the range [<code>result</code>, <code>result</code> + (<code>last</code> - <code>first</code>) ). The input and output sequences may coincide, resulting in an in-place transformation.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>transform</code> to compute the sum of two ranges using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...

int input1[6] = {-5,  0,  2,  3,  2,  4};
int input2[6] = { 3,  6, -2,  1,  2,  3};
int output[6];

thrust::plus<int> op;

thrust::transform(thrust::host, input1, input1 + 6, input2, output, op);

// output is now {-2,  6,  0,  4,  4,  7};
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator1's</code><code>value&#95;type</code> is convertible to <code>BinaryFunction's</code><code>first&#95;argument&#95;type</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>BinaryFunction's</code><code>second&#95;argument&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`BinaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a> and <code>BinaryFunction's</code><code>result&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the first input sequence. 
* **`last1`** The end of the first input sequence. 
* **`first2`** The beginning of the second input sequence. 
* **`result`** The beginning of the output sequence. 
* **`op`** The tranformation operation. 

**Preconditions**:
* <code>first1</code> may equal <code>result</code>, but the range <code>[first1, last1)</code> shall not overlap the range <code>[result, result + (last1 - first1))</code> otherwise. 
* <code>first2</code> may equal <code>result</code>, but the range <code>[first2, first2 + (last1 - first1))</code> shall not overlap the range <code>[result, result + (last1 - first1))</code> otherwise.

**Returns**:
The end of the output sequence.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/transform">https://en.cppreference.com/w/cpp/algorithm/transform</a>

<h3 id="function-transform">
Function <code>thrust::transform</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename BinaryFunction&gt;</span>
<span>OutputIterator </span><span><b>transform</b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;BinaryFunction op);</span></code>
This version of <code>transform</code> applies a binary function to each pair of elements from two input sequences and stores the result in the corresponding position in an output sequence. Specifically, for each iterator <code>i</code> in the range [<code>first1</code>, <code>last1</code>) and <code>j = first + (i - first1)</code> in the range [<code>first2</code>, <code>last2</code>) the operation <code>op(&#42;i,&#42;j)</code> is performed and the result is assigned to <code>&#42;o</code>, where <code>o</code> is the corresponding output iterator in the range [<code>result</code>, <code>result</code> + (<code>last</code> - <code>first</code>) ). The input and output sequences may coincide, resulting in an in-place transformation.


The following code snippet demonstrates how to use <code>transform</code>



```cpp
#include <thrust/transform.h>
#include <thrust/functional.h>

int input1[6] = {-5,  0,  2,  3,  2,  4};
int input2[6] = { 3,  6, -2,  1,  2,  3};
int output[6];

thrust::plus<int> op;

thrust::transform(input1, input1 + 6, input2, output, op);

// output is now {-2,  6,  0,  4,  4,  7};
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator1's</code><code>value&#95;type</code> is convertible to <code>BinaryFunction's</code><code>first&#95;argument&#95;type</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>BinaryFunction's</code><code>second&#95;argument&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`BinaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a> and <code>BinaryFunction's</code><code>result&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first1`** The beginning of the first input sequence. 
* **`last1`** The end of the first input sequence. 
* **`first2`** The beginning of the second input sequence. 
* **`result`** The beginning of the output sequence. 
* **`op`** The tranformation operation. 

**Preconditions**:
* <code>first1</code> may equal <code>result</code>, but the range <code>[first1, last1)</code> shall not overlap the range <code>[result, result + (last1 - first1))</code> otherwise. 
* <code>first2</code> may equal <code>result</code>, but the range <code>[first2, first2 + (last1 - first1))</code> shall not overlap the range <code>[result, result + (last1 - first1))</code> otherwise.

**Returns**:
The end of the output sequence.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/transform">https://en.cppreference.com/w/cpp/algorithm/transform</a>

<h3 id="function-transform-if">
Function <code>thrust::transform&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>transform_if</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;ForwardIterator result,</span>
<span>&nbsp;&nbsp;UnaryFunction op,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
This version of <code>transform&#95;if</code> conditionally applies a unary function to each element of an input sequence and stores the result in the corresponding position in an output sequence if the corresponding position in the input sequence satifies a predicate. Otherwise, the corresponding position in the output sequence is not modified.

Specifically, for each iterator <code>i</code> in the range <code>[first, last)</code> the predicate <code>pred(&#42;i)</code> is evaluated. If this predicate evaluates to <code>true</code>, the result of <code>op(&#42;i)</code> is assigned to <code>&#42;o</code>, where <code>o</code> is the corresponding output iterator in the range <code>[result, result + (last - first) )</code>. Otherwise, <code>op(&#42;i)</code> is not evaluated and no assignment occurs. The input and output sequences may coincide, resulting in an in-place transformation.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>transform&#95;if</code> to negate the odd-valued elements of a range using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...

int data[10]    = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};

struct is_odd
{
  __host__ __device__
  bool operator()(int x)
  {
    return x % 2;
  }
};

thrust::negate<int> op;
thrust::identity<int> identity;

// negate odd elements
thrust::transform_if(thrust::host, data, data + 10, data, op, is_odd()); // in-place transformation

// data is now {5, 0, 2, 3, 2, 4, 0, 1, 2, 8};
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>UnaryFunction's</code><code>argument&#95;type</code>. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`UnaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a> and <code>UnaryFunction's</code><code>result&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the input sequence. 
* **`last`** The end of the input sequence. 
* **`result`** The beginning of the output sequence. 
* **`op`** The tranformation operation. 
* **`pred`** The predicate operation. 

**Preconditions**:
<code>first</code> may equal <code>result</code>, but the range <code>[first, last)</code> shall not overlap the range <code>[result, result + (last - first))</code> otherwise.

**Returns**:
The end of the output sequence.

**See**:
thrust::transform 

<h3 id="function-transform-if">
Function <code>thrust::transform&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>ForwardIterator </span><span><b>transform_if</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;ForwardIterator result,</span>
<span>&nbsp;&nbsp;UnaryFunction op,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
This version of <code>transform&#95;if</code> conditionally applies a unary function to each element of an input sequence and stores the result in the corresponding position in an output sequence if the corresponding position in the input sequence satifies a predicate. Otherwise, the corresponding position in the output sequence is not modified.

Specifically, for each iterator <code>i</code> in the range <code>[first, last)</code> the predicate <code>pred(&#42;i)</code> is evaluated. If this predicate evaluates to <code>true</code>, the result of <code>op(&#42;i)</code> is assigned to <code>&#42;o</code>, where <code>o</code> is the corresponding output iterator in the range <code>[result, result + (last - first) )</code>. Otherwise, <code>op(&#42;i)</code> is not evaluated and no assignment occurs. The input and output sequences may coincide, resulting in an in-place transformation.


The following code snippet demonstrates how to use <code>transform&#95;if:</code>



```cpp
#include <thrust/transform.h>
#include <thrust/functional.h>

int data[10]    = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};

struct is_odd
{
  __host__ __device__
  bool operator()(int x)
  {
    return x % 2;
  }
};

thrust::negate<int> op;
thrust::identity<int> identity;

// negate odd elements
thrust::transform_if(data, data + 10, data, op, is_odd()); // in-place transformation

// data is now {5, 0, 2, 3, 2, 4, 0, 1, 2, 8};
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>UnaryFunction's</code><code>argument&#95;type</code>. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`UnaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a> and <code>UnaryFunction's</code><code>result&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first`** The beginning of the input sequence. 
* **`last`** The end of the input sequence. 
* **`result`** The beginning of the output sequence. 
* **`op`** The tranformation operation. 
* **`pred`** The predicate operation. 

**Preconditions**:
<code>first</code> may equal <code>result</code>, but the range <code>[first, last)</code> shall not overlap the range <code>[result, result + (last - first))</code> otherwise.

**Returns**:
The end of the output sequence.

**See**:
thrust::transform 

<h3 id="function-transform-if">
Function <code>thrust::transform&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>transform_if</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;ForwardIterator result,</span>
<span>&nbsp;&nbsp;UnaryFunction op,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
This version of <code>transform&#95;if</code> conditionally applies a unary function to each element of an input sequence and stores the result in the corresponding position in an output sequence if the corresponding position in a stencil sequence satisfies a predicate. Otherwise, the corresponding position in the output sequence is not modified.

Specifically, for each iterator <code>i</code> in the range <code>[first, last)</code> the predicate <code>pred(&#42;s)</code> is evaluated, where <code>s</code> is the corresponding input iterator in the range <code>[stencil, stencil + (last - first) )</code>. If this predicate evaluates to <code>true</code>, the result of <code>op(&#42;i)</code> is assigned to <code>&#42;o</code>, where <code>o</code> is the corresponding output iterator in the range <code>[result, result + (last - first) )</code>. Otherwise, <code>op(&#42;i)</code> is not evaluated and no assignment occurs. The input and output sequences may coincide, resulting in an in-place transformation.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>transform&#95;if</code> using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...

int data[10]    = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};
int stencil[10] = { 1, 0, 1,  0, 1, 0, 1,  0, 1, 0};

thrust::negate<int> op;
thrust::identity<int> identity;

thrust::transform_if(thrust::host, data, data + 10, stencil, data, op, identity); // in-place transformation

// data is now {5, 0, -2, -3, -2,  4, 0, -1, -2,  8};
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator1's</code><code>value&#95;type</code> is convertible to <code>UnaryFunction's</code><code>argument&#95;type</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`UnaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a> and <code>UnaryFunction's</code><code>result&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the input sequence. 
* **`last`** The end of the input sequence. 
* **`stencil`** The beginning of the stencil sequence. 
* **`result`** The beginning of the output sequence. 
* **`op`** The tranformation operation. 
* **`pred`** The predicate operation. 

**Preconditions**:
* <code>first</code> may equal <code>result</code>, but the range <code>[first, last)</code> shall not overlap the range <code>[result, result + (last - first))</code> otherwise. 
* <code>stencil</code> may equal <code>result</code>, but the range <code>[stencil, stencil + (last - first))</code> shall not overlap the range <code>[result, result + (last - first))</code> otherwise.

**Returns**:
The end of the output sequence.

**See**:
thrust::transform 

<h3 id="function-transform-if">
Function <code>thrust::transform&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>ForwardIterator </span><span><b>transform_if</b>(InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;ForwardIterator result,</span>
<span>&nbsp;&nbsp;UnaryFunction op,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
This version of <code>transform&#95;if</code> conditionally applies a unary function to each element of an input sequence and stores the result in the corresponding position in an output sequence if the corresponding position in a stencil sequence satisfies a predicate. Otherwise, the corresponding position in the output sequence is not modified.

Specifically, for each iterator <code>i</code> in the range <code>[first, last)</code> the predicate <code>pred(&#42;s)</code> is evaluated, where <code>s</code> is the corresponding input iterator in the range <code>[stencil, stencil + (last - first) )</code>. If this predicate evaluates to <code>true</code>, the result of <code>op(&#42;i)</code> is assigned to <code>&#42;o</code>, where <code>o</code> is the corresponding output iterator in the range <code>[result, result + (last - first) )</code>. Otherwise, <code>op(&#42;i)</code> is not evaluated and no assignment occurs. The input and output sequences may coincide, resulting in an in-place transformation.


The following code snippet demonstrates how to use <code>transform&#95;if:</code>



```cpp
#include <thrust/transform.h>
#include <thrust/functional.h>

int data[10]    = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};
int stencil[10] = { 1, 0, 1,  0, 1, 0, 1,  0, 1, 0};

thrust::negate<int> op;
thrust::identity<int> identity;

thrust::transform_if(data, data + 10, stencil, data, op, identity); // in-place transformation

// data is now {5, 0, -2, -3, -2,  4, 0, -1, -2,  8};
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator1's</code><code>value&#95;type</code> is convertible to <code>UnaryFunction's</code><code>argument&#95;type</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`UnaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a> and <code>UnaryFunction's</code><code>result&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first`** The beginning of the input sequence. 
* **`last`** The end of the input sequence. 
* **`stencil`** The beginning of the stencil sequence. 
* **`result`** The beginning of the output sequence. 
* **`op`** The tranformation operation. 
* **`pred`** The predicate operation. 

**Preconditions**:
* <code>first</code> may equal <code>result</code>, but the range <code>[first, last)</code> shall not overlap the range <code>[result, result + (last - first))</code> otherwise. 
* <code>stencil</code> may equal <code>result</code>, but the range <code>[stencil, stencil + (last - first))</code> shall not overlap the range <code>[result, result + (last - first))</code> otherwise.

**Returns**:
The end of the output sequence.

**See**:
thrust::transform 

<h3 id="function-transform-if">
Function <code>thrust::transform&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename BinaryFunction,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>transform_if</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator3 stencil,</span>
<span>&nbsp;&nbsp;ForwardIterator result,</span>
<span>&nbsp;&nbsp;BinaryFunction binary_op,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
This version of <code>transform&#95;if</code> conditionally applies a binary function to each pair of elements from two input sequences and stores the result in the corresponding position in an output sequence if the corresponding position in a stencil sequence satifies a predicate. Otherwise, the corresponding position in the output sequence is not modified.

Specifically, for each iterator <code>i</code> in the range <code>[first1, last1)</code> and <code>j = first2 + (i - first1)</code> in the range <code>[first2, first2 + (last1 - first1) )</code>, the predicate <code>pred(&#42;s)</code> is evaluated, where <code>s</code> is the corresponding input iterator in the range <code>[stencil, stencil + (last1 - first1) )</code>. If this predicate evaluates to <code>true</code>, the result of <code>binary&#95;op(&#42;i,&#42;j)</code> is assigned to <code>&#42;o</code>, where <code>o</code> is the corresponding output iterator in the range <code>[result, result + (last1 - first1) )</code>. Otherwise, <code>binary&#95;op(&#42;i,&#42;j)</code> is not evaluated and no assignment occurs. The input and output sequences may coincide, resulting in an in-place transformation.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>transform&#95;if</code> using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...

int input1[6]  = {-5,  0,  2,  3,  2,  4};
int input2[6]  = { 3,  6, -2,  1,  2,  3};
int stencil[8] = { 1,  0,  1,  0,  1,  0};
int output[6];

thrust::plus<int> op;
thrust::identity<int> identity;

thrust::transform_if(thrust::host, input1, input1 + 6, input2, stencil, output, op, identity);

// output is now {-2,  0,  0,  3,  4,  4};
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator1's</code><code>value&#95;type</code> is convertible to <code>BinaryFunction's</code><code>first&#95;argument&#95;type</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>BinaryFunction's</code><code>second&#95;argument&#95;type</code>. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`BinaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a> and <code>BinaryFunction's</code><code>result&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the first input sequence. 
* **`last1`** The end of the first input sequence. 
* **`first2`** The beginning of the second input sequence. 
* **`stencil`** The beginning of the stencil sequence. 
* **`result`** The beginning of the output sequence. 
* **`binary_op`** The transformation operation. 
* **`pred`** The predicate operation. 

**Preconditions**:
* <code>first1</code> may equal <code>result</code>, but the range <code>[first1, last1)</code> shall not overlap the range <code>[result, result + (last1 - first1))</code> otherwise. 
* <code>first2</code> may equal <code>result</code>, but the range <code>[first2, first2 + (last1 - first1))</code> shall not overlap the range <code>[result, result + (last1 - first1))</code> otherwise. 
* <code>stencil</code> may equal <code>result</code>, but the range <code>[stencil, stencil + (last1 - first1))</code> shall not overlap the range <code>[result, result + (last1 - first1))</code> otherwise.

**Returns**:
The end of the output sequence.

**See**:
thrust::transform 

<h3 id="function-transform-if">
Function <code>thrust::transform&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename BinaryFunction,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>ForwardIterator </span><span><b>transform_if</b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator3 stencil,</span>
<span>&nbsp;&nbsp;ForwardIterator result,</span>
<span>&nbsp;&nbsp;BinaryFunction binary_op,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
This version of <code>transform&#95;if</code> conditionally applies a binary function to each pair of elements from two input sequences and stores the result in the corresponding position in an output sequence if the corresponding position in a stencil sequence satifies a predicate. Otherwise, the corresponding position in the output sequence is not modified.

Specifically, for each iterator <code>i</code> in the range <code>[first1, last1)</code> and <code>j = first2 + (i - first1)</code> in the range <code>[first2, first2 + (last1 - first1) )</code>, the predicate <code>pred(&#42;s)</code> is evaluated, where <code>s</code> is the corresponding input iterator in the range <code>[stencil, stencil + (last1 - first1) )</code>. If this predicate evaluates to <code>true</code>, the result of <code>binary&#95;op(&#42;i,&#42;j)</code> is assigned to <code>&#42;o</code>, where <code>o</code> is the corresponding output iterator in the range <code>[result, result + (last1 - first1) )</code>. Otherwise, <code>binary&#95;op(&#42;i,&#42;j)</code> is not evaluated and no assignment occurs. The input and output sequences may coincide, resulting in an in-place transformation.


The following code snippet demonstrates how to use <code>transform&#95;if:</code>



```cpp
#include <thrust/transform.h>
#include <thrust/functional.h>

int input1[6]  = {-5,  0,  2,  3,  2,  4};
int input2[6]  = { 3,  6, -2,  1,  2,  3};
int stencil[8] = { 1,  0,  1,  0,  1,  0};
int output[6];

thrust::plus<int> op;
thrust::identity<int> identity;

thrust::transform_if(input1, input1 + 6, input2, stencil, output, op, identity);

// output is now {-2,  0,  0,  3,  4,  4};
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator1's</code><code>value&#95;type</code> is convertible to <code>BinaryFunction's</code><code>first&#95;argument&#95;type</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>BinaryFunction's</code><code>second&#95;argument&#95;type</code>. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>. 
* **`BinaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a> and <code>BinaryFunction's</code><code>result&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first1`** The beginning of the first input sequence. 
* **`last1`** The end of the first input sequence. 
* **`first2`** The beginning of the second input sequence. 
* **`stencil`** The beginning of the stencil sequence. 
* **`result`** The beginning of the output sequence. 
* **`binary_op`** The transformation operation. 
* **`pred`** The predicate operation. 

**Preconditions**:
* <code>first1</code> may equal <code>result</code>, but the range <code>[first1, last1)</code> shall not overlap the range <code>[result, result + (last1 - first1))</code> otherwise. 
* <code>first2</code> may equal <code>result</code>, but the range <code>[first2, first2 + (last1 - first1))</code> shall not overlap the range <code>[result, result + (last1 - first1))</code> otherwise. 
* <code>stencil</code> may equal <code>result</code>, but the range <code>[stencil, stencil + (last1 - first1))</code> shall not overlap the range <code>[result, result + (last1 - first1))</code> otherwise.

**Returns**:
The end of the output sequence.

**See**:
thrust::transform 


