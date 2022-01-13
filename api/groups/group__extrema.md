---
title: Extrema
parent: Reductions
grand_parent: Algorithms
nav_exclude: false
has_children: true
has_toc: false
---

# Extrema

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__extrema.html#function-min-element">thrust::min&#95;element</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span>
<br>
<span>template &lt;typename ForwardIterator&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__extrema.html#function-min-element">thrust::min&#95;element</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__extrema.html#function-min-element">thrust::min&#95;element</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;BinaryPredicate comp);</span>
<br>
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__extrema.html#function-min-element">thrust::min&#95;element</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;BinaryPredicate comp);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__extrema.html#function-max-element">thrust::max&#95;element</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span>
<br>
<span>template &lt;typename ForwardIterator&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__extrema.html#function-max-element">thrust::max&#95;element</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__extrema.html#function-max-element">thrust::max&#95;element</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;BinaryPredicate comp);</span>
<br>
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__extrema.html#function-max-element">thrust::max&#95;element</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;BinaryPredicate comp);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< ForwardIterator, ForwardIterator > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__extrema.html#function-minmax-element">thrust::minmax&#95;element</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span>
<br>
<span>template &lt;typename ForwardIterator&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< ForwardIterator, ForwardIterator > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__extrema.html#function-minmax-element">thrust::minmax&#95;element</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< ForwardIterator, ForwardIterator > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__extrema.html#function-minmax-element">thrust::minmax&#95;element</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;BinaryPredicate comp);</span>
<br>
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< ForwardIterator, ForwardIterator > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__extrema.html#function-minmax-element">thrust::minmax&#95;element</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;BinaryPredicate comp);</span>
</code>

## Functions

<h3 id="function-min-element">
Function <code>thrust::min&#95;element</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>min_element</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span></code>
<code>min&#95;element</code> finds the smallest element in the range <code>[first, last)</code>. It returns the first iterator <code>i</code> in <code>[first, last)</code> such that no other iterator in <code>[first, last)</code> points to a value smaller than <code>&#42;i</code>. The return value is <code>last</code> if and only if <code>[first, last)</code> is an empty range.

The two versions of <code>min&#95;element</code> differ in how they define whether one element is less than another. This version compares objects using <code>operator&lt;</code>. Specifically, this version of <code>min&#95;element</code> returns the first iterator <code>i</code> in <code>[first, last)</code> such that, for every iterator <code>j</code> in <code>[first, last)</code>, <code>&#42;j &lt; &#42;i</code> is <code>false</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.



```cpp
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
...
int data[6] = {1, 0, 2, 2, 1, 3};
int *result = thrust::min_element(thrust::host, data, data + 6);

// result is data + 1
// *result is 0
```

**Template Parameters**:
**`ForwardIterator`**: is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 

**Returns**:
An iterator pointing to the smallest element of the range <code>[first, last)</code>, if it is not an empty range; <code>last</code>, otherwise.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/min_element">https://en.cppreference.com/w/cpp/algorithm/min_element</a>

<h3 id="function-min-element">
Function <code>thrust::min&#95;element</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator&gt;</span>
<span>ForwardIterator </span><span><b>min_element</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span></code>
<code>min&#95;element</code> finds the smallest element in the range <code>[first, last)</code>. It returns the first iterator <code>i</code> in <code>[first, last)</code> such that no other iterator in <code>[first, last)</code> points to a value smaller than <code>&#42;i</code>. The return value is <code>last</code> if and only if <code>[first, last)</code> is an empty range.

The two versions of <code>min&#95;element</code> differ in how they define whether one element is less than another. This version compares objects using <code>operator&lt;</code>. Specifically, this version of <code>min&#95;element</code> returns the first iterator <code>i</code> in <code>[first, last)</code> such that, for every iterator <code>j</code> in <code>[first, last)</code>, <code>&#42;j &lt; &#42;i</code> is <code>false</code>.



```cpp
#include <thrust/extrema.h>
...
int data[6] = {1, 0, 2, 2, 1, 3};
int *result = thrust::min_element(data, data + 6);

// result is data + 1
// *result is 0
```

**Template Parameters**:
**`ForwardIterator`**: is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 

**Returns**:
An iterator pointing to the smallest element of the range <code>[first, last)</code>, if it is not an empty range; <code>last</code>, otherwise.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/min_element">https://en.cppreference.com/w/cpp/algorithm/min_element</a>

<h3 id="function-min-element">
Function <code>thrust::min&#95;element</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>min_element</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;BinaryPredicate comp);</span></code>
<code>min&#95;element</code> finds the smallest element in the range <code>[first, last)</code>. It returns the first iterator <code>i</code> in <code>[first, last)</code> such that no other iterator in <code>[first, last)</code> points to a value smaller than <code>&#42;i</code>. The return value is <code>last</code> if and only if <code>[first, last)</code> is an empty range.

The two versions of <code>min&#95;element</code> differ in how they define whether one element is less than another. This version compares objects using a function object <code>comp</code>. Specifically, this version of <code>min&#95;element</code> returns the first iterator <code>i</code> in <code>[first, last)</code> such that, for every iterator <code>j</code> in <code>[first, last)</code>, <code>comp(&#42;j, &#42;i)</code> is <code>false</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>min&#95;element</code> to find the smallest element of a collection of key-value pairs using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
...

struct key_value
{
  int key;
  int value;
};

struct compare_key_value
{
  __host__ __device__
  bool operator()(key_value lhs, key_value rhs)
  {
    return lhs.key < rhs.key;
  }
};

...
key_value data[4] = { {4,5}, {0,7}, {2,3}, {6,1} };

key_value *smallest = thrust::min_element(thrust::host, data, data + 4, compare_key_value());

// smallest == data + 1
// *smallest == {0,7}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to both <code>comp's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`comp`** A binary predicate used for comparison. 

**Returns**:
An iterator pointing to the smallest element of the range <code>[first, last)</code>, if it is not an empty range; <code>last</code>, otherwise.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/min_element">https://en.cppreference.com/w/cpp/algorithm/min_element</a>

<h3 id="function-min-element">
Function <code>thrust::min&#95;element</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>ForwardIterator </span><span><b>min_element</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;BinaryPredicate comp);</span></code>
<code>min&#95;element</code> finds the smallest element in the range <code>[first, last)</code>. It returns the first iterator <code>i</code> in <code>[first, last)</code> such that no other iterator in <code>[first, last)</code> points to a value smaller than <code>&#42;i</code>. The return value is <code>last</code> if and only if <code>[first, last)</code> is an empty range.

The two versions of <code>min&#95;element</code> differ in how they define whether one element is less than another. This version compares objects using a function object <code>comp</code>. Specifically, this version of <code>min&#95;element</code> returns the first iterator <code>i</code> in <code>[first, last)</code> such that, for every iterator <code>j</code> in <code>[first, last)</code>, <code>comp(&#42;j, &#42;i)</code> is <code>false</code>.


The following code snippet demonstrates how to use <code>min&#95;element</code> to find the smallest element of a collection of key-value pairs.



```cpp
#include <thrust/extrema.h>

struct key_value
{
  int key;
  int value;
};

struct compare_key_value
{
  __host__ __device__
  bool operator()(key_value lhs, key_value rhs)
  {
    return lhs.key < rhs.key;
  }
};

...
key_value data[4] = { {4,5}, {0,7}, {2,3}, {6,1} };

key_value *smallest = thrust::min_element(data, data + 4, compare_key_value());

// smallest == data + 1
// *smallest == {0,7}
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to both <code>comp's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`comp`** A binary predicate used for comparison. 

**Returns**:
An iterator pointing to the smallest element of the range <code>[first, last)</code>, if it is not an empty range; <code>last</code>, otherwise.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/min_element">https://en.cppreference.com/w/cpp/algorithm/min_element</a>

<h3 id="function-max-element">
Function <code>thrust::max&#95;element</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>max_element</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span></code>
<code>max&#95;element</code> finds the largest element in the range <code>[first, last)</code>. It returns the first iterator <code>i</code> in <code>[first, last)</code> such that no other iterator in <code>[first, last)</code> points to a value larger than <code>&#42;i</code>. The return value is <code>last</code> if and only if <code>[first, last)</code> is an empty range.

The two versions of <code>max&#95;element</code> differ in how they define whether one element is greater than another. This version compares objects using <code>operator&lt;</code>. Specifically, this version of <code>max&#95;element</code> returns the first iterator <code>i</code> in <code>[first, last)</code> such that, for every iterator <code>j</code> in <code>[first, last)</code>, <code>&#42;i &lt; &#42;j</code> is <code>false</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.



```cpp
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
...
int data[6] = {1, 0, 2, 2, 1, 3};
int *result = thrust::max_element(thrust::host, data, data + 6);

// *result == 3
```

**Template Parameters**:
* **`A`** Thrust backend system. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 

**Returns**:
An iterator pointing to the largest element of the range <code>[first, last)</code>, if it is not an empty range; <code>last</code>, otherwise.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/max_element">https://en.cppreference.com/w/cpp/algorithm/max_element</a>

<h3 id="function-max-element">
Function <code>thrust::max&#95;element</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator&gt;</span>
<span>ForwardIterator </span><span><b>max_element</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span></code>
<code>max&#95;element</code> finds the largest element in the range <code>[first, last)</code>. It returns the first iterator <code>i</code> in <code>[first, last)</code> such that no other iterator in <code>[first, last)</code> points to a value larger than <code>&#42;i</code>. The return value is <code>last</code> if and only if <code>[first, last)</code> is an empty range.

The two versions of <code>max&#95;element</code> differ in how they define whether one element is greater than another. This version compares objects using <code>operator&lt;</code>. Specifically, this version of <code>max&#95;element</code> returns the first iterator <code>i</code> in <code>[first, last)</code> such that, for every iterator <code>j</code> in <code>[first, last)</code>, <code>&#42;i &lt; &#42;j</code> is <code>false</code>.



```cpp
#include <thrust/extrema.h>
...
int data[6] = {1, 0, 2, 2, 1, 3};
int *result = thrust::max_element(data, data + 6);

// *result == 3
```

**Template Parameters**:
**`ForwardIterator`**: is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 

**Returns**:
An iterator pointing to the largest element of the range <code>[first, last)</code>, if it is not an empty range; <code>last</code>, otherwise.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/max_element">https://en.cppreference.com/w/cpp/algorithm/max_element</a>

<h3 id="function-max-element">
Function <code>thrust::max&#95;element</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>max_element</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;BinaryPredicate comp);</span></code>
<code>max&#95;element</code> finds the largest element in the range <code>[first, last)</code>. It returns the first iterator <code>i</code> in <code>[first, last)</code> such that no other iterator in <code>[first, last)</code> points to a value larger than <code>&#42;i</code>. The return value is <code>last</code> if and only if <code>[first, last)</code> is an empty range.

The two versions of <code>max&#95;element</code> differ in how they define whether one element is less than another. This version compares objects using a function object <code>comp</code>. Specifically, this version of <code>max&#95;element</code> returns the first iterator <code>i</code> in <code>[first, last)</code> such that, for every iterator <code>j</code> in <code>[first, last)</code>, <code>comp(&#42;i, &#42;j)</code> is <code>false</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>max&#95;element</code> to find the largest element of a collection of key-value pairs using the <code>thrust::host</code> execution policy for parallelization.



```cpp
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
...

struct key_value
{
  int key;
  int value;
};

struct compare_key_value
{
  __host__ __device__
  bool operator()(key_value lhs, key_value rhs)
  {
    return lhs.key < rhs.key;
  }
};

...
key_value data[4] = { {4,5}, {0,7}, {2,3}, {6,1} };

key_value *largest = thrust::max_element(thrust::host, data, data + 4, compare_key_value());

// largest == data + 3
// *largest == {6,1}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to both <code>comp's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`comp`** A binary predicate used for comparison. 

**Returns**:
An iterator pointing to the largest element of the range <code>[first, last)</code>, if it is not an empty range; <code>last</code>, otherwise.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/max_element">https://en.cppreference.com/w/cpp/algorithm/max_element</a>

<h3 id="function-max-element">
Function <code>thrust::max&#95;element</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>ForwardIterator </span><span><b>max_element</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;BinaryPredicate comp);</span></code>
<code>max&#95;element</code> finds the largest element in the range <code>[first, last)</code>. It returns the first iterator <code>i</code> in <code>[first, last)</code> such that no other iterator in <code>[first, last)</code> points to a value larger than <code>&#42;i</code>. The return value is <code>last</code> if and only if <code>[first, last)</code> is an empty range.

The two versions of <code>max&#95;element</code> differ in how they define whether one element is less than another. This version compares objects using a function object <code>comp</code>. Specifically, this version of <code>max&#95;element</code> returns the first iterator <code>i</code> in <code>[first, last)</code> such that, for every iterator <code>j</code> in <code>[first, last)</code>, <code>comp(&#42;i, &#42;j)</code> is <code>false</code>.


The following code snippet demonstrates how to use <code>max&#95;element</code> to find the largest element of a collection of key-value pairs.



```cpp
#include <thrust/extrema.h>

struct key_value
{
  int key;
  int value;
};

struct compare_key_value
{
  __host__ __device__
  bool operator()(key_value lhs, key_value rhs)
  {
    return lhs.key < rhs.key;
  }
};

...
key_value data[4] = { {4,5}, {0,7}, {2,3}, {6,1} };

key_value *largest = thrust::max_element(data, data + 4, compare_key_value());

// largest == data + 3
// *largest == {6,1}
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to both <code>comp's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`comp`** A binary predicate used for comparison. 

**Returns**:
An iterator pointing to the largest element of the range <code>[first, last)</code>, if it is not an empty range; <code>last</code>, otherwise.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/max_element">https://en.cppreference.com/w/cpp/algorithm/max_element</a>

<h3 id="function-minmax-element">
Function <code>thrust::minmax&#95;element</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< ForwardIterator, ForwardIterator > </span><span><b>minmax_element</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span></code>
<code>minmax&#95;element</code> finds the smallest and largest elements in the range <code>[first, last)</code>. It returns a pair of iterators <code>(imin, imax)</code> where <code>imin</code> is the same iterator returned by <code>min&#95;element</code> and <code>imax</code> is the same iterator returned by <code>max&#95;element</code>. This function is potentially more efficient than separate calls to <code>min&#95;element</code> and <code>max&#95;element</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.



```cpp
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
...
int data[6] = {1, 0, 2, 2, 1, 3};
thrust::pair<int *, int *> result = thrust::minmax_element(thrust::host, data, data + 6);

// result.first is data + 1
// result.second is data + 5
// *result.first is 0
// *result.second is 3
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 

**Returns**:
A pair of iterator pointing to the smallest and largest elements of the range <code>[first, last)</code>, if it is not an empty range; <code>last</code>, otherwise.

**See**:
* min_element 
* max_element 
* <a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1840.pdf">http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1840.pdf</a>

<h3 id="function-minmax-element">
Function <code>thrust::minmax&#95;element</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< ForwardIterator, ForwardIterator > </span><span><b>minmax_element</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span></code>
<code>minmax&#95;element</code> finds the smallest and largest elements in the range <code>[first, last)</code>. It returns a pair of iterators <code>(imin, imax)</code> where <code>imin</code> is the same iterator returned by <code>min&#95;element</code> and <code>imax</code> is the same iterator returned by <code>max&#95;element</code>. This function is potentially more efficient than separate calls to <code>min&#95;element</code> and <code>max&#95;element</code>.



```cpp
#include <thrust/extrema.h>
...
int data[6] = {1, 0, 2, 2, 1, 3};
thrust::pair<int *, int *> result = thrust::minmax_element(data, data + 6);

// result.first is data + 1
// result.second is data + 5
// *result.first is 0
// *result.second is 3
```

**Template Parameters**:
**`ForwardIterator`**: is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 

**Returns**:
A pair of iterator pointing to the smallest and largest elements of the range <code>[first, last)</code>, if it is not an empty range; <code>last</code>, otherwise.

**See**:
* min_element 
* max_element 
* <a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1840.pdf">http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1840.pdf</a>

<h3 id="function-minmax-element">
Function <code>thrust::minmax&#95;element</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< ForwardIterator, ForwardIterator > </span><span><b>minmax_element</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;BinaryPredicate comp);</span></code>
<code>minmax&#95;element</code> finds the smallest and largest elements in the range <code>[first, last)</code>. It returns a pair of iterators <code>(imin, imax)</code> where <code>imin</code> is the same iterator returned by <code>min&#95;element</code> and <code>imax</code> is the same iterator returned by <code>max&#95;element</code>. This function is potentially more efficient than separate calls to <code>min&#95;element</code> and <code>max&#95;element</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>minmax&#95;element</code> to find the smallest and largest elements of a collection of key-value pairs using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/execution_policy.h>
...

struct key_value
{
  int key;
  int value;
};

struct compare_key_value
{
  __host__ __device__
  bool operator()(key_value lhs, key_value rhs)
  {
    return lhs.key < rhs.key;
  }
};

...
key_value data[4] = { {4,5}, {0,7}, {2,3}, {6,1} };

thrust::pair<key_value*,key_value*> extrema = thrust::minmax_element(thrust::host, data, data + 4, compare_key_value());

// extrema.first   == data + 1
// *extrema.first  == {0,7}
// extrema.second  == data + 3
// *extrema.second == {6,1}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to both <code>comp's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`comp`** A binary predicate used for comparison. 

**Returns**:
A pair of iterator pointing to the smallest and largest elements of the range <code>[first, last)</code>, if it is not an empty range; <code>last</code>, otherwise.

**See**:
* min_element 
* max_element 
* <a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1840.pdf">http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1840.pdf</a>

<h3 id="function-minmax-element">
Function <code>thrust::minmax&#95;element</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< ForwardIterator, ForwardIterator > </span><span><b>minmax_element</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;BinaryPredicate comp);</span></code>
<code>minmax&#95;element</code> finds the smallest and largest elements in the range <code>[first, last)</code>. It returns a pair of iterators <code>(imin, imax)</code> where <code>imin</code> is the same iterator returned by <code>min&#95;element</code> and <code>imax</code> is the same iterator returned by <code>max&#95;element</code>. This function is potentially more efficient than separate calls to <code>min&#95;element</code> and <code>max&#95;element</code>.


The following code snippet demonstrates how to use <code>minmax&#95;element</code> to find the smallest and largest elements of a collection of key-value pairs.



```cpp
#include <thrust/extrema.h>
#include <thrust/pair.h>

struct key_value
{
  int key;
  int value;
};

struct compare_key_value
{
  __host__ __device__
  bool operator()(key_value lhs, key_value rhs)
  {
    return lhs.key < rhs.key;
  }
};

...
key_value data[4] = { {4,5}, {0,7}, {2,3}, {6,1} };

thrust::pair<key_value*,key_value*> extrema = thrust::minmax_element(data, data + 4, compare_key_value());

// extrema.first   == data + 1
// *extrema.first  == {0,7}
// extrema.second  == data + 3
// *extrema.second == {6,1}
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to both <code>comp's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`comp`** A binary predicate used for comparison. 

**Returns**:
A pair of iterator pointing to the smallest and largest elements of the range <code>[first, last)</code>, if it is not an empty range; <code>last</code>, otherwise.

**See**:
* min_element 
* max_element 
* <a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1840.pdf">http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1840.pdf</a>


