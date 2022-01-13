---
title: Comparisons
parent: Reductions
grand_parent: Algorithms
nav_exclude: false
has_children: true
has_toc: false
---

# Comparisons

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__comparisons.html#function-equal">thrust::equal</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2&gt;</span>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__comparisons.html#function-equal">thrust::equal</a></b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__comparisons.html#function-equal">thrust::equal</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__comparisons.html#function-equal">thrust::equal</a></b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span>
</code>

## Functions

<h3 id="function-equal">
Function <code>thrust::equal</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2&gt;</span>
<span>__host__ __device__ bool </span><span><b>equal</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2);</span></code>
<code>equal</code> returns <code>true</code> if the two ranges <code>[first1, last1)</code> and <code>[first2, first2 + (last1 - first1))</code> are identical when compared element-by-element, and otherwise returns <code>false</code>.

This version of <code>equal</code> returns <code>true</code> if and only if for every iterator <code>i</code> in <code>[first1, last1)</code>, <code>&#42;i == &#42;(first2 + (i - first1))</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>equal</code> to test two ranges for equality using the <code>thrust::host</code> execution policy:



```cpp
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
...
int A1[7] = {3, 1, 4, 1, 5, 9, 3};
int A2[7] = {3, 1, 4, 2, 8, 5, 7};
...
bool result = thrust::equal(thrust::host, A1, A1 + 7, A2);

// result == false
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>, and <code>InputIterator1's</code><code>value&#95;type</code> can be compared for equality with <code>InputIterator2's</code><code>value&#95;type</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>, and <code>InputIterator2's</code><code>value&#95;type</code> can be compared for equality with <code>InputIterator1's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the first sequence. 
* **`last1`** The end of the first sequence. 
* **`first2`** The beginning of the second sequence. 

**Returns**:
<code>true</code>, if the sequences are equal; <code>false</code>, otherwise.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/equal">https://en.cppreference.com/w/cpp/algorithm/equal</a>

<h3 id="function-equal">
Function <code>thrust::equal</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2&gt;</span>
<span>bool </span><span><b>equal</b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2);</span></code>
<code>equal</code> returns <code>true</code> if the two ranges <code>[first1, last1)</code> and <code>[first2, first2 + (last1 - first1))</code> are identical when compared element-by-element, and otherwise returns <code>false</code>.

This version of <code>equal</code> returns <code>true</code> if and only if for every iterator <code>i</code> in <code>[first1, last1)</code>, <code>&#42;i == &#42;(first2 + (i - first1))</code>.


The following code snippet demonstrates how to use <code>equal</code> to test two ranges for equality.



```cpp
#include <thrust/equal.h>
...
int A1[7] = {3, 1, 4, 1, 5, 9, 3};
int A2[7] = {3, 1, 4, 2, 8, 5, 7};
...
bool result = thrust::equal(A1, A1 + 7, A2);

// result == false
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>, and <code>InputIterator1's</code><code>value&#95;type</code> can be compared for equality with <code>InputIterator2's</code><code>value&#95;type</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>, and <code>InputIterator2's</code><code>value&#95;type</code> can be compared for equality with <code>InputIterator1's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first1`** The beginning of the first sequence. 
* **`last1`** The end of the first sequence. 
* **`first2`** The beginning of the second sequence. 

**Returns**:
<code>true</code>, if the sequences are equal; <code>false</code>, otherwise.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/equal">https://en.cppreference.com/w/cpp/algorithm/equal</a>

<h3 id="function-equal">
Function <code>thrust::equal</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ bool </span><span><b>equal</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span></code>
<code>equal</code> returns <code>true</code> if the two ranges <code>[first1, last1)</code> and <code>[first2, first2 + (last1 - first1))</code> are identical when compared element-by-element, and otherwise returns <code>false</code>.

This version of <code>equal</code> returns <code>true</code> if and only if for every iterator <code>i</code> in <code>[first1, last1)</code>, <code>binary&#95;pred(&#42;i, &#42;(first2 + (i - first1)))</code> is <code>true</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>equal</code> to compare the elements in two ranges modulo 2 using the <code>thrust::host</code> execution policy.



```cpp
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
...

struct compare_modulo_two
{
  __host__ __device__
  bool operator()(int x, int y) const
  {
    return (x % 2) == (y % 2);
  }
};
...
int x[6] = {0, 2, 4, 6, 8, 10};
int y[6] = {1, 3, 5, 7, 9, 11};

bool result = thrust::equal(x, x + 6, y, compare_modulo_two());

// result is false
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator1's</code><code>value&#95;type</code> is convertible to <code>BinaryPredicate's</code><code>first&#95;argument&#95;type</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>BinaryPredicate's</code><code>second&#95;argument&#95;type</code>. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the first sequence. 
* **`last1`** The end of the first sequence. 
* **`first2`** The beginning of the second sequence. 
* **`binary_pred`** Binary predicate used to test element equality. 

**Returns**:
<code>true</code>, if the sequences are equal; <code>false</code>, otherwise.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/equal">https://en.cppreference.com/w/cpp/algorithm/equal</a>

<h3 id="function-equal">
Function <code>thrust::equal</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>bool </span><span><b>equal</b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span></code>
<code>equal</code> returns <code>true</code> if the two ranges <code>[first1, last1)</code> and <code>[first2, first2 + (last1 - first1))</code> are identical when compared element-by-element, and otherwise returns <code>false</code>.

This version of <code>equal</code> returns <code>true</code> if and only if for every iterator <code>i</code> in <code>[first1, last1)</code>, <code>binary&#95;pred(&#42;i, &#42;(first2 + (i - first1)))</code> is <code>true</code>.


The following code snippet demonstrates how to use <code>equal</code> to compare the elements in two ranges modulo 2.



```cpp
#include <thrust/equal.h>

struct compare_modulo_two
{
  __host__ __device__
  bool operator()(int x, int y) const
  {
    return (x % 2) == (y % 2);
  }
};
...
int x[6] = {0, 2, 4, 6, 8, 10};
int y[6] = {1, 3, 5, 7, 9, 11};

bool result = thrust::equal(x, x + 5, y, compare_modulo_two());

// result is true
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator1's</code><code>value&#95;type</code> is convertible to <code>BinaryPredicate's</code><code>first&#95;argument&#95;type</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>BinaryPredicate's</code><code>second&#95;argument&#95;type</code>. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.

**Function Parameters**:
* **`first1`** The beginning of the first sequence. 
* **`last1`** The end of the first sequence. 
* **`first2`** The beginning of the second sequence. 
* **`binary_pred`** Binary predicate used to test element equality. 

**Returns**:
<code>true</code>, if the sequences are equal; <code>false</code>, otherwise.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/equal">https://en.cppreference.com/w/cpp/algorithm/equal</a>


