---
title: Predicates
parent: Reductions
grand_parent: Algorithms
nav_exclude: false
has_children: true
has_toc: false
---

# Predicates

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__predicates.html#function-is-partitioned">thrust::is&#95;partitioned</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__predicates.html#function-is-partitioned">thrust::is&#95;partitioned</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__predicates.html#function-is-sorted">thrust::is&#95;sorted</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span>
<br>
<span>template &lt;typename ForwardIterator&gt;</span>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__predicates.html#function-is-sorted">thrust::is&#95;sorted</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Compare&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__predicates.html#function-is-sorted">thrust::is&#95;sorted</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Compare comp);</span>
<br>
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Compare&gt;</span>
<span>bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__predicates.html#function-is-sorted">thrust::is&#95;sorted</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Compare comp);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__predicates.html#function-is-sorted-until">thrust::is&#95;sorted&#95;until</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span>
<br>
<span>template &lt;typename ForwardIterator&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__predicates.html#function-is-sorted-until">thrust::is&#95;sorted&#95;until</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Compare&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__predicates.html#function-is-sorted-until">thrust::is&#95;sorted&#95;until</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Compare comp);</span>
<br>
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Compare&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__predicates.html#function-is-sorted-until">thrust::is&#95;sorted&#95;until</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Compare comp);</span>
</code>

## Functions

<h3 id="function-is-partitioned">
Function <code>thrust::is&#95;partitioned</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ bool </span><span><b>is_partitioned</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>is&#95;partitioned</code> returns <code>true</code> if the given range is partitioned with respect to a predicate, and <code>false</code> otherwise.

Specifically, <code>is&#95;partitioned</code> returns <code>true</code> if <code>[first, last)</code> is empty of if <code>[first, last)</code> is partitioned by <code>pred</code>, i.e. if all elements that satisfy <code>pred</code> appear before those that do not.

The algorithm's execution is parallelized as determined by <code>exec</code>.



```cpp
#include <thrust/partition.h>
#include <thrust/execution_policy.h>

struct is_even
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return (x % 2) == 0;
  }
};

...

int A[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
int B[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

thrust::is_partitioned(thrust::host, A, A + 10, is_even()); // returns true
thrust::is_partitioned(thrust::host, B, B + 10, is_even()); // returns false
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the range to consider. 
* **`last`** The end of the range to consider. 
* **`pred`** A function object which decides to which partition each element of the range <code>[first, last)</code> belongs. 

**Returns**:
<code>true</code> if the range <code>[first, last)</code> is partitioned with respect to <code>pred</code>, or if <code>[first, last)</code> is empty. <code>false</code>, otherwise.

**See**:
<code>partition</code>

<h3 id="function-is-partitioned">
Function <code>thrust::is&#95;partitioned</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>bool </span><span><b>is_partitioned</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>is&#95;partitioned</code> returns <code>true</code> if the given range is partitioned with respect to a predicate, and <code>false</code> otherwise.

Specifically, <code>is&#95;partitioned</code> returns <code>true</code> if <code>[first, last)</code> is empty of if <code>[first, last)</code> is partitioned by <code>pred</code>, i.e. if all elements that satisfy <code>pred</code> appear before those that do not.



```cpp
#include <thrust/partition.h>

struct is_even
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return (x % 2) == 0;
  }
};

...

int A[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
int B[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

thrust::is_partitioned(A, A + 10, is_even()); // returns true
thrust::is_partitioned(B, B + 10, is_even()); // returns false
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first`** The beginning of the range to consider. 
* **`last`** The end of the range to consider. 
* **`pred`** A function object which decides to which partition each element of the range <code>[first, last)</code> belongs. 

**Returns**:
<code>true</code> if the range <code>[first, last)</code> is partitioned with respect to <code>pred</code>, or if <code>[first, last)</code> is empty. <code>false</code>, otherwise.

**See**:
<code>partition</code>

<h3 id="function-is-sorted">
Function <code>thrust::is&#95;sorted</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator&gt;</span>
<span>__host__ __device__ bool </span><span><b>is_sorted</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span></code>
<code>is&#95;sorted</code> returns <code>true</code> if the range <code>[first, last)</code> is sorted in ascending order, and <code>false</code> otherwise.

Specifically, this version of <code>is&#95;sorted</code> returns <code>false</code> if for some iterator <code>i</code> in the range <code>[first, last - 1)</code> the expression <code>&#42;(i + 1) &lt; &#42;i</code> is <code>true</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code demonstrates how to use <code>is&#95;sorted</code> to test whether the contents of a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device&#95;vector</a></code> are stored in ascending order using the <code>thrust::device</code> execution policy for parallelization:



```cpp
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
...
thrust::device_vector<int> v(6);
v[0] = 1;
v[1] = 4;
v[2] = 2;
v[3] = 8;
v[4] = 5;
v[5] = 7;

bool result = thrust::is_sorted(thrust::device, v.begin(), v.end());

// result == false

thrust::sort(v.begin(), v.end());
result = thrust::is_sorted(thrust::device, v.begin(), v.end());

// result == true
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, <code>ForwardIterator's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, and the ordering on objects of <code>ForwardIterator's</code><code>value&#95;type</code> is a _strict weak ordering_, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 

**Returns**:
<code>true</code>, if the sequence is sorted; <code>false</code>, otherwise.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/is_sorted">https://en.cppreference.com/w/cpp/algorithm/is_sorted</a>
* is_sorted_until 
* <code>sort</code>
* <code>stable&#95;sort</code>
* <code>less&lt;T&gt;</code>

<h3 id="function-is-sorted">
Function <code>thrust::is&#95;sorted</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator&gt;</span>
<span>bool </span><span><b>is_sorted</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span></code>
<code>is&#95;sorted</code> returns <code>true</code> if the range <code>[first, last)</code> is sorted in ascending order, and <code>false</code> otherwise.

Specifically, this version of <code>is&#95;sorted</code> returns <code>false</code> if for some iterator <code>i</code> in the range <code>[first, last - 1)</code> the expression <code>&#42;(i + 1) &lt; &#42;i</code> is <code>true</code>.


The following code demonstrates how to use <code>is&#95;sorted</code> to test whether the contents of a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device&#95;vector</a></code> are stored in ascending order.



```cpp
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
...
thrust::device_vector<int> v(6);
v[0] = 1;
v[1] = 4;
v[2] = 2;
v[3] = 8;
v[4] = 5;
v[5] = 7;

bool result = thrust::is_sorted(v.begin(), v.end());

// result == false

thrust::sort(v.begin(), v.end());
result = thrust::is_sorted(v.begin(), v.end());

// result == true
```

**Template Parameters**:
**`ForwardIterator`**: is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, <code>ForwardIterator's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, and the ordering on objects of <code>ForwardIterator's</code><code>value&#95;type</code> is a _strict weak ordering_, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 

**Returns**:
<code>true</code>, if the sequence is sorted; <code>false</code>, otherwise.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/is_sorted">https://en.cppreference.com/w/cpp/algorithm/is_sorted</a>
* is_sorted_until 
* <code>sort</code>
* <code>stable&#95;sort</code>
* <code>less&lt;T&gt;</code>

<h3 id="function-is-sorted">
Function <code>thrust::is&#95;sorted</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Compare&gt;</span>
<span>__host__ __device__ bool </span><span><b>is_sorted</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Compare comp);</span></code>
<code>is&#95;sorted</code> returns <code>true</code> if the range <code>[first, last)</code> is sorted in ascending order accoring to a user-defined comparison operation, and <code>false</code> otherwise.

Specifically, this version of <code>is&#95;sorted</code> returns <code>false</code> if for some iterator <code>i</code> in the range <code>[first, last - 1)</code> the expression <code>comp(&#42;(i + 1), &#42;i)</code> is <code>true</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>is&#95;sorted</code> to test whether the contents of a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device&#95;vector</a></code> are stored in descending order using the <code>thrust::device</code> execution policy for parallelization:



```cpp
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
...
thrust::device_vector<int> v(6);
v[0] = 1;
v[1] = 4;
v[2] = 2;
v[3] = 8;
v[4] = 5;
v[5] = 7;

thrust::greater<int> comp;
bool result = thrust::is_sorted(thrust::device, v.begin(), v.end(), comp);

// result == false

thrust::sort(v.begin(), v.end(), comp);
result = thrust::is_sorted(thrust::device, v.begin(), v.end(), comp);

// result == true
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to both <code>StrictWeakOrdering's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>. 
* **`Compare`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`comp`** Comparison operator. 

**Returns**:
<code>true</code>, if the sequence is sorted according to comp; <code>false</code>, otherwise.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/is_sorted">https://en.cppreference.com/w/cpp/algorithm/is_sorted</a>
* <code>sort</code>
* <code>stable&#95;sort</code>
* <code>less&lt;T&gt;</code>

<h3 id="function-is-sorted">
Function <code>thrust::is&#95;sorted</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Compare&gt;</span>
<span>bool </span><span><b>is_sorted</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Compare comp);</span></code>
<code>is&#95;sorted</code> returns <code>true</code> if the range <code>[first, last)</code> is sorted in ascending order accoring to a user-defined comparison operation, and <code>false</code> otherwise.

Specifically, this version of <code>is&#95;sorted</code> returns <code>false</code> if for some iterator <code>i</code> in the range <code>[first, last - 1)</code> the expression <code>comp(&#42;(i + 1), &#42;i)</code> is <code>true</code>.


The following code snippet demonstrates how to use <code>is&#95;sorted</code> to test whether the contents of a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device&#95;vector</a></code> are stored in descending order.



```cpp
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
...
thrust::device_vector<int> v(6);
v[0] = 1;
v[1] = 4;
v[2] = 2;
v[3] = 8;
v[4] = 5;
v[5] = 7;

thrust::greater<int> comp;
bool result = thrust::is_sorted(v.begin(), v.end(), comp);

// result == false

thrust::sort(v.begin(), v.end(), comp);
result = thrust::is_sorted(v.begin(), v.end(), comp);

// result == true
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to both <code>StrictWeakOrdering's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>. 
* **`Compare`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`comp`** Comparison operator. 

**Returns**:
<code>true</code>, if the sequence is sorted according to comp; <code>false</code>, otherwise.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/is_sorted">https://en.cppreference.com/w/cpp/algorithm/is_sorted</a>
* <code>sort</code>
* <code>stable&#95;sort</code>
* <code>less&lt;T&gt;</code>

<h3 id="function-is-sorted-until">
Function <code>thrust::is&#95;sorted&#95;until</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>is_sorted_until</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span></code>
This version of <code>is&#95;sorted&#95;until</code> returns the last iterator <code>i</code> in <code>[first,last]</code> for which the range <code>[first,last)</code> is sorted using <code>operator&lt;</code>. If <code>distance(first,last) &lt; 2</code>, <code>is&#95;sorted&#95;until</code> simply returns <code>last</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>is&#95;sorted&#95;until</code> to find the first position in an array where the data becomes unsorted using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

...
 
int A[8] = {0, 1, 2, 3, 0, 1, 2, 3};

int * B = thrust::is_sorted_until(thrust::host, A, A + 8);

// B - A is 4
// [A, B) is sorted
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a> and <code>ForwardIterator's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the range of interest. 
* **`last`** The end of the range of interest. 

**Returns**:
The last iterator in the input range for which it is sorted.

**See**:
* <code>is&#95;sorted</code>
* <code>sort</code>
* <code>sort&#95;by&#95;key</code>
* <code>stable&#95;sort</code>
* <code>stable&#95;sort&#95;by&#95;key</code>

<h3 id="function-is-sorted-until">
Function <code>thrust::is&#95;sorted&#95;until</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator&gt;</span>
<span>ForwardIterator </span><span><b>is_sorted_until</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span></code>
This version of <code>is&#95;sorted&#95;until</code> returns the last iterator <code>i</code> in <code>[first,last]</code> for which the range <code>[first,last)</code> is sorted using <code>operator&lt;</code>. If <code>distance(first,last) &lt; 2</code>, <code>is&#95;sorted&#95;until</code> simply returns <code>last</code>.


The following code snippet demonstrates how to use <code>is&#95;sorted&#95;until</code> to find the first position in an array where the data becomes unsorted:



```cpp
#include <thrust/sort.h>

...
 
int A[8] = {0, 1, 2, 3, 0, 1, 2, 3};

int * B = thrust::is_sorted_until(A, A + 8);

// B - A is 4
// [A, B) is sorted
```

**Template Parameters**:
**`ForwardIterator`**: is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a> and <code>ForwardIterator's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.

**Function Parameters**:
* **`first`** The beginning of the range of interest. 
* **`last`** The end of the range of interest. 

**Returns**:
The last iterator in the input range for which it is sorted.

**See**:
* <code>is&#95;sorted</code>
* <code>sort</code>
* <code>sort&#95;by&#95;key</code>
* <code>stable&#95;sort</code>
* <code>stable&#95;sort&#95;by&#95;key</code>

<h3 id="function-is-sorted-until">
Function <code>thrust::is&#95;sorted&#95;until</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Compare&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>is_sorted_until</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Compare comp);</span></code>
This version of <code>is&#95;sorted&#95;until</code> returns the last iterator <code>i</code> in <code>[first,last]</code> for which the range <code>[first,last)</code> is sorted using the function object <code>comp</code>. If <code>distance(first,last) &lt; 2</code>, <code>is&#95;sorted&#95;until</code> simply returns <code>last</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>is&#95;sorted&#95;until</code> to find the first position in an array where the data becomes unsorted in descending order using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

...
 
int A[8] = {3, 2, 1, 0, 3, 2, 1, 0};

thrust::greater<int> comp;
int * B = thrust::is_sorted_until(thrust::host, A, A + 8, comp);

// B - A is 4
// [A, B) is sorted in descending order
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a> and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to <code>Compare's</code><code>argument&#95;type</code>. 
* **`Compare`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization: 
* **`first`** The beginning of the range of interest. 
* **`last`** The end of the range of interest. 
* **`comp`** The function object to use for comparison. 

**Returns**:
The last iterator in the input range for which it is sorted.

**See**:
* <code>is&#95;sorted</code>
* <code>sort</code>
* <code>sort&#95;by&#95;key</code>
* <code>stable&#95;sort</code>
* <code>stable&#95;sort&#95;by&#95;key</code>

<h3 id="function-is-sorted-until">
Function <code>thrust::is&#95;sorted&#95;until</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Compare&gt;</span>
<span>ForwardIterator </span><span><b>is_sorted_until</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Compare comp);</span></code>
This version of <code>is&#95;sorted&#95;until</code> returns the last iterator <code>i</code> in <code>[first,last]</code> for which the range <code>[first,last)</code> is sorted using the function object <code>comp</code>. If <code>distance(first,last) &lt; 2</code>, <code>is&#95;sorted&#95;until</code> simply returns <code>last</code>.


The following code snippet demonstrates how to use <code>is&#95;sorted&#95;until</code> to find the first position in an array where the data becomes unsorted in descending order:



```cpp
#include <thrust/sort.h>
#include <thrust/functional.h>

...
 
int A[8] = {3, 2, 1, 0, 3, 2, 1, 0};

thrust::greater<int> comp;
int * B = thrust::is_sorted_until(A, A + 8, comp);

// B - A is 4
// [A, B) is sorted in descending order
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a> and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to <code>Compare's</code><code>argument&#95;type</code>. 
* **`Compare`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`first`** The beginning of the range of interest. 
* **`last`** The end of the range of interest. 
* **`comp`** The function object to use for comparison. 

**Returns**:
The last iterator in the input range for which it is sorted.

**See**:
* <code>is&#95;sorted</code>
* <code>sort</code>
* <code>sort&#95;by&#95;key</code>
* <code>stable&#95;sort</code>
* <code>stable&#95;sort&#95;by&#95;key</code>


