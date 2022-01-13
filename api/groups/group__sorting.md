---
title: Sorting
parent: Algorithms
grand_parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Sorting

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__sorting.html#function-sort">thrust::sort</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;RandomAccessIterator first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator last);</span>
<br>
<span>template &lt;typename RandomAccessIterator&gt;</span>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__sorting.html#function-sort">thrust::sort</a></b>(RandomAccessIterator first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator last);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__sorting.html#function-sort">thrust::sort</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;RandomAccessIterator first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator last,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span>
<br>
<span>template &lt;typename RandomAccessIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__sorting.html#function-sort">thrust::sort</a></b>(RandomAccessIterator first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator last,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__sorting.html#function-stable-sort">thrust::stable&#95;sort</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;RandomAccessIterator first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator last);</span>
<br>
<span>template &lt;typename RandomAccessIterator&gt;</span>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__sorting.html#function-stable-sort">thrust::stable&#95;sort</a></b>(RandomAccessIterator first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator last);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__sorting.html#function-stable-sort">thrust::stable&#95;sort</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;RandomAccessIterator first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator last,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span>
<br>
<span>template &lt;typename RandomAccessIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__sorting.html#function-stable-sort">thrust::stable&#95;sort</a></b>(RandomAccessIterator first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator last,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator1,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator2&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__sorting.html#function-sort-by-key">thrust::sort&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;RandomAccessIterator1 keys_first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator1 keys_last,</span>
<span>&nbsp;&nbsp;RandomAccessIterator2 values_first);</span>
<br>
<span>template &lt;typename RandomAccessIterator1,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator2&gt;</span>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__sorting.html#function-sort-by-key">thrust::sort&#95;by&#95;key</a></b>(RandomAccessIterator1 keys_first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator1 keys_last,</span>
<span>&nbsp;&nbsp;RandomAccessIterator2 values_first);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator1,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__sorting.html#function-sort-by-key">thrust::sort&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;RandomAccessIterator1 keys_first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator1 keys_last,</span>
<span>&nbsp;&nbsp;RandomAccessIterator2 values_first,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span>
<br>
<span>template &lt;typename RandomAccessIterator1,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__sorting.html#function-sort-by-key">thrust::sort&#95;by&#95;key</a></b>(RandomAccessIterator1 keys_first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator1 keys_last,</span>
<span>&nbsp;&nbsp;RandomAccessIterator2 values_first,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator1,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator2&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__sorting.html#function-stable-sort-by-key">thrust::stable&#95;sort&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;RandomAccessIterator1 keys_first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator1 keys_last,</span>
<span>&nbsp;&nbsp;RandomAccessIterator2 values_first);</span>
<br>
<span>template &lt;typename RandomAccessIterator1,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator2&gt;</span>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__sorting.html#function-stable-sort-by-key">thrust::stable&#95;sort&#95;by&#95;key</a></b>(RandomAccessIterator1 keys_first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator1 keys_last,</span>
<span>&nbsp;&nbsp;RandomAccessIterator2 values_first);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator1,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__sorting.html#function-stable-sort-by-key">thrust::stable&#95;sort&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;RandomAccessIterator1 keys_first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator1 keys_last,</span>
<span>&nbsp;&nbsp;RandomAccessIterator2 values_first,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span>
<br>
<span>template &lt;typename RandomAccessIterator1,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__sorting.html#function-stable-sort-by-key">thrust::stable&#95;sort&#95;by&#95;key</a></b>(RandomAccessIterator1 keys_first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator1 keys_last,</span>
<span>&nbsp;&nbsp;RandomAccessIterator2 values_first,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span>
</code>

## Functions

<h3 id="function-sort">
Function <code>thrust::sort</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator&gt;</span>
<span>__host__ __device__ void </span><span><b>sort</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;RandomAccessIterator first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator last);</span></code>
<code>sort</code> sorts the elements in <code>[first, last)</code> into ascending order, meaning that if <code>i</code> and <code>j</code> are any two valid iterators in <code>[first, last)</code> such that <code>i</code> precedes <code>j</code>, then <code>&#42;j</code> is not less than <code>&#42;i</code>. Note: <code>sort</code> is not guaranteed to be stable. That is, suppose that <code>&#42;i</code> and <code>&#42;j</code> are equivalent: neither one is less than the other. It is not guaranteed that the relative order of these two elements will be preserved by <code>sort</code>.

This version of <code>sort</code> compares objects using <code>operator&lt;</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>sort</code> to sort a sequence of integers using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
...
const int N = 6;
int A[N] = {1, 4, 2, 8, 5, 7};
thrust::sort(thrust::host, A, A + N);
// A is now {1, 2, 4, 5, 7, 8}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`RandomAccessIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>, <code>RandomAccessIterator</code> is mutable, and <code>RandomAccessIterator's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, and the ordering relation on <code>RandomAccessIterator's</code><code>value&#95;type</code> is a _strict weak ordering_, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/sort">https://en.cppreference.com/w/cpp/algorithm/sort</a>
* <code>stable&#95;sort</code>
* <code>sort&#95;by&#95;key</code>

<h3 id="function-sort">
Function <code>thrust::sort</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RandomAccessIterator&gt;</span>
<span>void </span><span><b>sort</b>(RandomAccessIterator first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator last);</span></code>
<code>sort</code> sorts the elements in <code>[first, last)</code> into ascending order, meaning that if <code>i</code> and <code>j</code> are any two valid iterators in <code>[first, last)</code> such that <code>i</code> precedes <code>j</code>, then <code>&#42;j</code> is not less than <code>&#42;i</code>. Note: <code>sort</code> is not guaranteed to be stable. That is, suppose that <code>&#42;i</code> and <code>&#42;j</code> are equivalent: neither one is less than the other. It is not guaranteed that the relative order of these two elements will be preserved by <code>sort</code>.

This version of <code>sort</code> compares objects using <code>operator&lt;</code>.


The following code snippet demonstrates how to use <code>sort</code> to sort a sequence of integers.



```cpp
#include <thrust/sort.h>
...
const int N = 6;
int A[N] = {1, 4, 2, 8, 5, 7};
thrust::sort(A, A + N);
// A is now {1, 2, 4, 5, 7, 8}
```

**Template Parameters**:
**`RandomAccessIterator`**: is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>, <code>RandomAccessIterator</code> is mutable, and <code>RandomAccessIterator's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, and the ordering relation on <code>RandomAccessIterator's</code><code>value&#95;type</code> is a _strict weak ordering_, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/sort">https://en.cppreference.com/w/cpp/algorithm/sort</a>
* <code>stable&#95;sort</code>
* <code>sort&#95;by&#95;key</code>

<h3 id="function-sort">
Function <code>thrust::sort</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>__host__ __device__ void </span><span><b>sort</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;RandomAccessIterator first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator last,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span></code>
<code>sort</code> sorts the elements in <code>[first, last)</code> into ascending order, meaning that if <code>i</code> and <code>j</code> are any two valid iterators in <code>[first, last)</code> such that <code>i</code> precedes <code>j</code>, then <code>&#42;j</code> is not less than <code>&#42;i</code>. Note: <code>sort</code> is not guaranteed to be stable. That is, suppose that <code>&#42;i</code> and <code>&#42;j</code> are equivalent: neither one is less than the other. It is not guaranteed that the relative order of these two elements will be preserved by <code>sort</code>.

This version of <code>sort</code> compares objects using a function object <code>comp</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code demonstrates how to sort integers in descending order using the greater<int> comparison operator using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...
const int N = 6;
int A[N] = {1, 4, 2, 8, 5, 7};
thrust::sort(thrust::host, A, A + N, thrust::greater<int>());
// A is now {8, 7, 5, 4, 2, 1};
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`RandomAccessIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>, <code>RandomAccessIterator</code> is mutable, and <code>RandomAccessIterator's</code><code>value&#95;type</code> is convertible to <code>StrictWeakOrdering's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>. 
* **`StrictWeakOrdering`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`comp`** Comparison operator.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/sort">https://en.cppreference.com/w/cpp/algorithm/sort</a>
* <code>stable&#95;sort</code>
* <code>sort&#95;by&#95;key</code>

<h3 id="function-sort">
Function <code>thrust::sort</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RandomAccessIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>__host__ __device__ void </span><span><b>sort</b>(RandomAccessIterator first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator last,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span></code>
<code>sort</code> sorts the elements in <code>[first, last)</code> into ascending order, meaning that if <code>i</code> and <code>j</code> are any two valid iterators in <code>[first, last)</code> such that <code>i</code> precedes <code>j</code>, then <code>&#42;j</code> is not less than <code>&#42;i</code>. Note: <code>sort</code> is not guaranteed to be stable. That is, suppose that <code>&#42;i</code> and <code>&#42;j</code> are equivalent: neither one is less than the other. It is not guaranteed that the relative order of these two elements will be preserved by <code>sort</code>.

This version of <code>sort</code> compares objects using a function object <code>comp</code>.


The following code demonstrates how to sort integers in descending order using the greater<int> comparison operator.



```cpp
#include <thrust/sort.h>
#include <thrust/functional.h>
...
const int N = 6;
int A[N] = {1, 4, 2, 8, 5, 7};
thrust::sort(A, A + N, thrust::greater<int>());
// A is now {8, 7, 5, 4, 2, 1};
```

**Template Parameters**:
* **`RandomAccessIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>, <code>RandomAccessIterator</code> is mutable, and <code>RandomAccessIterator's</code><code>value&#95;type</code> is convertible to <code>StrictWeakOrdering's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>. 
* **`StrictWeakOrdering`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`comp`** Comparison operator.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/sort">https://en.cppreference.com/w/cpp/algorithm/sort</a>
* <code>stable&#95;sort</code>
* <code>sort&#95;by&#95;key</code>

<h3 id="function-stable-sort">
Function <code>thrust::stable&#95;sort</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator&gt;</span>
<span>__host__ __device__ void </span><span><b>stable_sort</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;RandomAccessIterator first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator last);</span></code>
<code>stable&#95;sort</code> is much like <code>sort:</code> it sorts the elements in <code>[first, last)</code> into ascending order, meaning that if <code>i</code> and <code>j</code> are any two valid iterators in <code>[first, last)</code> such that <code>i</code> precedes <code>j</code>, then <code>&#42;j</code> is not less than <code>&#42;i</code>.

As the name suggests, <code>stable&#95;sort</code> is stable: it preserves the relative ordering of equivalent elements. That is, if <code>x</code> and <code>y</code> are elements in <code>[first, last)</code> such that <code>x</code> precedes <code>y</code>, and if the two elements are equivalent (neither <code>x &lt; y</code> nor <code>y &lt; x</code>) then a postcondition of <code>stable&#95;sort</code> is that <code>x</code> still precedes <code>y</code>.

This version of <code>stable&#95;sort</code> compares objects using <code>operator&lt;</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>sort</code> to sort a sequence of integers using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
...
const int N = 6;
int A[N] = {1, 4, 2, 8, 5, 7};
thrust::stable_sort(thrust::host, A, A + N);
// A is now {1, 2, 4, 5, 7, 8}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`RandomAccessIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>, <code>RandomAccessIterator</code> is mutable, and <code>RandomAccessIterator's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, and the ordering relation on <code>RandomAccessIterator's</code><code>value&#95;type</code> is a _strict weak ordering_, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/stable_sort">https://en.cppreference.com/w/cpp/algorithm/stable_sort</a>
* <code>sort</code>
* <code>stable&#95;sort&#95;by&#95;key</code>

<h3 id="function-stable-sort">
Function <code>thrust::stable&#95;sort</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RandomAccessIterator&gt;</span>
<span>void </span><span><b>stable_sort</b>(RandomAccessIterator first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator last);</span></code>
<code>stable&#95;sort</code> is much like <code>sort:</code> it sorts the elements in <code>[first, last)</code> into ascending order, meaning that if <code>i</code> and <code>j</code> are any two valid iterators in <code>[first, last)</code> such that <code>i</code> precedes <code>j</code>, then <code>&#42;j</code> is not less than <code>&#42;i</code>.

As the name suggests, <code>stable&#95;sort</code> is stable: it preserves the relative ordering of equivalent elements. That is, if <code>x</code> and <code>y</code> are elements in <code>[first, last)</code> such that <code>x</code> precedes <code>y</code>, and if the two elements are equivalent (neither <code>x &lt; y</code> nor <code>y &lt; x</code>) then a postcondition of <code>stable&#95;sort</code> is that <code>x</code> still precedes <code>y</code>.

This version of <code>stable&#95;sort</code> compares objects using <code>operator&lt;</code>.


The following code snippet demonstrates how to use <code>sort</code> to sort a sequence of integers.



```cpp
#include <thrust/sort.h>
...
const int N = 6;
int A[N] = {1, 4, 2, 8, 5, 7};
thrust::stable_sort(A, A + N);
// A is now {1, 2, 4, 5, 7, 8}
```

**Template Parameters**:
**`RandomAccessIterator`**: is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>, <code>RandomAccessIterator</code> is mutable, and <code>RandomAccessIterator's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, and the ordering relation on <code>RandomAccessIterator's</code><code>value&#95;type</code> is a _strict weak ordering_, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements.

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/stable_sort">https://en.cppreference.com/w/cpp/algorithm/stable_sort</a>
* <code>sort</code>
* <code>stable&#95;sort&#95;by&#95;key</code>

<h3 id="function-stable-sort">
Function <code>thrust::stable&#95;sort</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>__host__ __device__ void </span><span><b>stable_sort</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;RandomAccessIterator first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator last,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span></code>
<code>stable&#95;sort</code> is much like <code>sort:</code> it sorts the elements in <code>[first, last)</code> into ascending order, meaning that if <code>i</code> and <code>j</code> are any two valid iterators in <code>[first, last)</code> such that <code>i</code> precedes <code>j</code>, then <code>&#42;j</code> is not less than <code>&#42;i</code>.

As the name suggests, <code>stable&#95;sort</code> is stable: it preserves the relative ordering of equivalent elements. That is, if <code>x</code> and <code>y</code> are elements in <code>[first, last)</code> such that <code>x</code> precedes <code>y</code>, and if the two elements are equivalent (neither <code>x &lt; y</code> nor <code>y &lt; x</code>) then a postcondition of <code>stable&#95;sort</code> is that <code>x</code> still precedes <code>y</code>.

This version of <code>stable&#95;sort</code> compares objects using a function object <code>comp</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code demonstrates how to sort integers in descending order using the greater<int> comparison operator using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...
const int N = 6;
int A[N] = {1, 4, 2, 8, 5, 7};
thrust::sort(A, A + N, thrust::greater<int>());
// A is now {8, 7, 5, 4, 2, 1};
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`RandomAccessIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>, <code>RandomAccessIterator</code> is mutable, and <code>RandomAccessIterator's</code><code>value&#95;type</code> is convertible to <code>StrictWeakOrdering's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>. 
* **`StrictWeakOrdering`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`comp`** Comparison operator.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/stable_sort">https://en.cppreference.com/w/cpp/algorithm/stable_sort</a>
* <code>sort</code>
* <code>stable&#95;sort&#95;by&#95;key</code>

<h3 id="function-stable-sort">
Function <code>thrust::stable&#95;sort</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RandomAccessIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>void </span><span><b>stable_sort</b>(RandomAccessIterator first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator last,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span></code>
<code>stable&#95;sort</code> is much like <code>sort:</code> it sorts the elements in <code>[first, last)</code> into ascending order, meaning that if <code>i</code> and <code>j</code> are any two valid iterators in <code>[first, last)</code> such that <code>i</code> precedes <code>j</code>, then <code>&#42;j</code> is not less than <code>&#42;i</code>.

As the name suggests, <code>stable&#95;sort</code> is stable: it preserves the relative ordering of equivalent elements. That is, if <code>x</code> and <code>y</code> are elements in <code>[first, last)</code> such that <code>x</code> precedes <code>y</code>, and if the two elements are equivalent (neither <code>x &lt; y</code> nor <code>y &lt; x</code>) then a postcondition of <code>stable&#95;sort</code> is that <code>x</code> still precedes <code>y</code>.

This version of <code>stable&#95;sort</code> compares objects using a function object <code>comp</code>.


The following code demonstrates how to sort integers in descending order using the greater<int> comparison operator.



```cpp
#include <thrust/sort.h>
#include <thrust/functional.h>
...
const int N = 6;
int A[N] = {1, 4, 2, 8, 5, 7};
thrust::sort(A, A + N, thrust::greater<int>());
// A is now {8, 7, 5, 4, 2, 1};
```

**Template Parameters**:
* **`RandomAccessIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>, <code>RandomAccessIterator</code> is mutable, and <code>RandomAccessIterator's</code><code>value&#95;type</code> is convertible to <code>StrictWeakOrdering's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>. 
* **`StrictWeakOrdering`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`comp`** Comparison operator.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/stable_sort">https://en.cppreference.com/w/cpp/algorithm/stable_sort</a>
* <code>sort</code>
* <code>stable&#95;sort&#95;by&#95;key</code>

<h3 id="function-sort-by-key">
Function <code>thrust::sort&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator1,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator2&gt;</span>
<span>__host__ __device__ void </span><span><b>sort_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;RandomAccessIterator1 keys_first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator1 keys_last,</span>
<span>&nbsp;&nbsp;RandomAccessIterator2 values_first);</span></code>
<code>sort&#95;by&#95;key</code> performs a key-value sort. That is, <code>sort&#95;by&#95;key</code> sorts the elements in <code>[keys&#95;first, keys&#95;last)</code> and <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> into ascending key order, meaning that if <code>i</code> and <code>j</code> are any two valid iterators in <code>[keys&#95;first, keys&#95;last)</code> such that <code>i</code> precedes <code>j</code>, and <code>p</code> and <code>q</code> are iterators in <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> corresponding to <code>i</code> and <code>j</code> respectively, then <code>&#42;j</code> is not less than <code>&#42;i</code>.

Note: <code>sort&#95;by&#95;key</code> is not guaranteed to be stable. That is, suppose that <code>&#42;i</code> and <code>&#42;j</code> are equivalent: neither one is less than the other. It is not guaranteed that the relative order of these two keys or the relative order of their corresponding values will be preserved by <code>sort&#95;by&#95;key</code>.

This version of <code>sort&#95;by&#95;key</code> compares key objects using <code>operator&lt;</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>sort&#95;by&#95;key</code> to sort an array of character values using integers as sorting keys using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
...
const int N = 6;
int    keys[N] = {  1,   4,   2,   8,   5,   7};
char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};
thrust::sort_by_key(thrust::host, keys, keys + N, values);
// keys is now   {  1,   2,   4,   5,   7,   8}
// values is now {'a', 'c', 'b', 'e', 'f', 'd'}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`RandomAccessIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>, <code>RandomAccessIterator1</code> is mutable, and <code>RandomAccessIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, and the ordering relation on <code>RandomAccessIterator1's</code><code>value&#95;type</code> is a _strict weak ordering_, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements. 
* **`RandomAccessIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator">Random Access Iterator</a>, and <code>RandomAccessIterator2</code> is mutable.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`keys_first`** The beginning of the key sequence. 
* **`keys_last`** The end of the key sequence. 
* **`values_first`** The beginning of the value sequence.

**Preconditions**:
The range <code>[keys&#95;first, keys&#95;last))</code> shall not overlap the range <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/sort">https://en.cppreference.com/w/cpp/algorithm/sort</a>
* <code>stable&#95;sort&#95;by&#95;key</code>
* <code>sort</code>

<h3 id="function-sort-by-key">
Function <code>thrust::sort&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RandomAccessIterator1,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator2&gt;</span>
<span>void </span><span><b>sort_by_key</b>(RandomAccessIterator1 keys_first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator1 keys_last,</span>
<span>&nbsp;&nbsp;RandomAccessIterator2 values_first);</span></code>
<code>sort&#95;by&#95;key</code> performs a key-value sort. That is, <code>sort&#95;by&#95;key</code> sorts the elements in <code>[keys&#95;first, keys&#95;last)</code> and <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> into ascending key order, meaning that if <code>i</code> and <code>j</code> are any two valid iterators in <code>[keys&#95;first, keys&#95;last)</code> such that <code>i</code> precedes <code>j</code>, and <code>p</code> and <code>q</code> are iterators in <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> corresponding to <code>i</code> and <code>j</code> respectively, then <code>&#42;j</code> is not less than <code>&#42;i</code>.

Note: <code>sort&#95;by&#95;key</code> is not guaranteed to be stable. That is, suppose that <code>&#42;i</code> and <code>&#42;j</code> are equivalent: neither one is less than the other. It is not guaranteed that the relative order of these two keys or the relative order of their corresponding values will be preserved by <code>sort&#95;by&#95;key</code>.

This version of <code>sort&#95;by&#95;key</code> compares key objects using <code>operator&lt;</code>.


The following code snippet demonstrates how to use <code>sort&#95;by&#95;key</code> to sort an array of character values using integers as sorting keys.



```cpp
#include <thrust/sort.h>
...
const int N = 6;
int    keys[N] = {  1,   4,   2,   8,   5,   7};
char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};
thrust::sort_by_key(keys, keys + N, values);
// keys is now   {  1,   2,   4,   5,   7,   8}
// values is now {'a', 'c', 'b', 'e', 'f', 'd'}
```

**Template Parameters**:
* **`RandomAccessIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>, <code>RandomAccessIterator1</code> is mutable, and <code>RandomAccessIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, and the ordering relation on <code>RandomAccessIterator1's</code><code>value&#95;type</code> is a _strict weak ordering_, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements. 
* **`RandomAccessIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator">Random Access Iterator</a>, and <code>RandomAccessIterator2</code> is mutable.

**Function Parameters**:
* **`keys_first`** The beginning of the key sequence. 
* **`keys_last`** The end of the key sequence. 
* **`values_first`** The beginning of the value sequence.

**Preconditions**:
The range <code>[keys&#95;first, keys&#95;last))</code> shall not overlap the range <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/sort">https://en.cppreference.com/w/cpp/algorithm/sort</a>
* <code>stable&#95;sort&#95;by&#95;key</code>
* <code>sort</code>

<h3 id="function-sort-by-key">
Function <code>thrust::sort&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator1,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>__host__ __device__ void </span><span><b>sort_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;RandomAccessIterator1 keys_first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator1 keys_last,</span>
<span>&nbsp;&nbsp;RandomAccessIterator2 values_first,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span></code>
<code>sort&#95;by&#95;key</code> performs a key-value sort. That is, <code>sort&#95;by&#95;key</code> sorts the elements in <code>[keys&#95;first, keys&#95;last)</code> and <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> into ascending key order, meaning that if <code>i</code> and <code>j</code> are any two valid iterators in <code>[keys&#95;first, keys&#95;last)</code> such that <code>i</code> precedes <code>j</code>, and <code>p</code> and <code>q</code> are iterators in <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> corresponding to <code>i</code> and <code>j</code> respectively, then <code>&#42;j</code> is not less than <code>&#42;i</code>.

Note: <code>sort&#95;by&#95;key</code> is not guaranteed to be stable. That is, suppose that <code>&#42;i</code> and <code>&#42;j</code> are equivalent: neither one is less than the other. It is not guaranteed that the relative order of these two keys or the relative order of their corresponding values will be preserved by <code>sort&#95;by&#95;key</code>.

This version of <code>sort&#95;by&#95;key</code> compares key objects using a function object <code>comp</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>sort&#95;by&#95;key</code> to sort an array of character values using integers as sorting keys using the <code>thrust::host</code> execution policy for parallelization.The keys are sorted in descending order using the <code>greater&lt;int&gt;</code> comparison operator.



```cpp
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
...
const int N = 6;
int    keys[N] = {  1,   4,   2,   8,   5,   7};
char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};
thrust::sort_by_key(thrust::host, keys, keys + N, values, thrust::greater<int>());
// keys is now   {  8,   7,   5,   4,   2,   1}
// values is now {'d', 'f', 'e', 'b', 'c', 'a'}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`RandomAccessIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>, <code>RandomAccessIterator1</code> is mutable, and <code>RandomAccessIterator1's</code><code>value&#95;type</code> is convertible to <code>StrictWeakOrdering's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>. 
* **`RandomAccessIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator">Random Access Iterator</a>, and <code>RandomAccessIterator2</code> is mutable. 
* **`StrictWeakOrdering`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`keys_first`** The beginning of the key sequence. 
* **`keys_last`** The end of the key sequence. 
* **`values_first`** The beginning of the value sequence. 
* **`comp`** Comparison operator.

**Preconditions**:
The range <code>[keys&#95;first, keys&#95;last))</code> shall not overlap the range <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/sort">https://en.cppreference.com/w/cpp/algorithm/sort</a>
* <code>stable&#95;sort&#95;by&#95;key</code>
* <code>sort</code>

<h3 id="function-sort-by-key">
Function <code>thrust::sort&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RandomAccessIterator1,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>void </span><span><b>sort_by_key</b>(RandomAccessIterator1 keys_first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator1 keys_last,</span>
<span>&nbsp;&nbsp;RandomAccessIterator2 values_first,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span></code>
<code>sort&#95;by&#95;key</code> performs a key-value sort. That is, <code>sort&#95;by&#95;key</code> sorts the elements in <code>[keys&#95;first, keys&#95;last)</code> and <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> into ascending key order, meaning that if <code>i</code> and <code>j</code> are any two valid iterators in <code>[keys&#95;first, keys&#95;last)</code> such that <code>i</code> precedes <code>j</code>, and <code>p</code> and <code>q</code> are iterators in <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> corresponding to <code>i</code> and <code>j</code> respectively, then <code>&#42;j</code> is not less than <code>&#42;i</code>.

Note: <code>sort&#95;by&#95;key</code> is not guaranteed to be stable. That is, suppose that <code>&#42;i</code> and <code>&#42;j</code> are equivalent: neither one is less than the other. It is not guaranteed that the relative order of these two keys or the relative order of their corresponding values will be preserved by <code>sort&#95;by&#95;key</code>.

This version of <code>sort&#95;by&#95;key</code> compares key objects using a function object <code>comp</code>.


The following code snippet demonstrates how to use <code>sort&#95;by&#95;key</code> to sort an array of character values using integers as sorting keys. The keys are sorted in descending order using the greater<int> comparison operator.



```cpp
#include <thrust/sort.h>
...
const int N = 6;
int    keys[N] = {  1,   4,   2,   8,   5,   7};
char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};
thrust::sort_by_key(keys, keys + N, values, thrust::greater<int>());
// keys is now   {  8,   7,   5,   4,   2,   1}
// values is now {'d', 'f', 'e', 'b', 'c', 'a'}
```

**Template Parameters**:
* **`RandomAccessIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>, <code>RandomAccessIterator1</code> is mutable, and <code>RandomAccessIterator1's</code><code>value&#95;type</code> is convertible to <code>StrictWeakOrdering's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>. 
* **`RandomAccessIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator">Random Access Iterator</a>, and <code>RandomAccessIterator2</code> is mutable. 
* **`StrictWeakOrdering`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`keys_first`** The beginning of the key sequence. 
* **`keys_last`** The end of the key sequence. 
* **`values_first`** The beginning of the value sequence. 
* **`comp`** Comparison operator.

**Preconditions**:
The range <code>[keys&#95;first, keys&#95;last))</code> shall not overlap the range <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/sort">https://en.cppreference.com/w/cpp/algorithm/sort</a>
* <code>stable&#95;sort&#95;by&#95;key</code>
* <code>sort</code>

<h3 id="function-stable-sort-by-key">
Function <code>thrust::stable&#95;sort&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator1,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator2&gt;</span>
<span>__host__ __device__ void </span><span><b>stable_sort_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;RandomAccessIterator1 keys_first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator1 keys_last,</span>
<span>&nbsp;&nbsp;RandomAccessIterator2 values_first);</span></code>
<code>stable&#95;sort&#95;by&#95;key</code> performs a key-value sort. That is, <code>stable&#95;sort&#95;by&#95;key</code> sorts the elements in <code>[keys&#95;first, keys&#95;last)</code> and <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> into ascending key order, meaning that if <code>i</code> and <code>j</code> are any two valid iterators in <code>[keys&#95;first, keys&#95;last)</code> such that <code>i</code> precedes <code>j</code>, and <code>p</code> and <code>q</code> are iterators in <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> corresponding to <code>i</code> and <code>j</code> respectively, then <code>&#42;j</code> is not less than <code>&#42;i</code>.

As the name suggests, <code>stable&#95;sort&#95;by&#95;key</code> is stable: it preserves the relative ordering of equivalent elements. That is, if <code>x</code> and <code>y</code> are elements in <code>[keys&#95;first, keys&#95;last)</code> such that <code>x</code> precedes <code>y</code>, and if the two elements are equivalent (neither <code>x &lt; y</code> nor <code>y &lt; x</code>) then a postcondition of <code>stable&#95;sort&#95;by&#95;key</code> is that <code>x</code> still precedes <code>y</code>.

This version of <code>stable&#95;sort&#95;by&#95;key</code> compares key objects using <code>operator&lt;</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>stable&#95;sort&#95;by&#95;key</code> to sort an array of characters using integers as sorting keys using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
...
const int N = 6;
int    keys[N] = {  1,   4,   2,   8,   5,   7};
char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};
thrust::stable_sort_by_key(thrust::host, keys, keys + N, values);
// keys is now   {  1,   2,   4,   5,   7,   8}
// values is now {'a', 'c', 'b', 'e', 'f', 'd'}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`RandomAccessIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>, <code>RandomAccessIterator1</code> is mutable, and <code>RandomAccessIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, and the ordering relation on <code>RandomAccessIterator1's</code><code>value&#95;type</code> is a _strict weak ordering_, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements. 
* **`RandomAccessIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator">Random Access Iterator</a>, and <code>RandomAccessIterator2</code> is mutable.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`keys_first`** The beginning of the key sequence. 
* **`keys_last`** The end of the key sequence. 
* **`values_first`** The beginning of the value sequence.

**Preconditions**:
The range <code>[keys&#95;first, keys&#95;last))</code> shall not overlap the range <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/sort">https://en.cppreference.com/w/cpp/algorithm/sort</a>
* <code>sort&#95;by&#95;key</code>
* <code>stable&#95;sort</code>

<h3 id="function-stable-sort-by-key">
Function <code>thrust::stable&#95;sort&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RandomAccessIterator1,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator2&gt;</span>
<span>void </span><span><b>stable_sort_by_key</b>(RandomAccessIterator1 keys_first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator1 keys_last,</span>
<span>&nbsp;&nbsp;RandomAccessIterator2 values_first);</span></code>
<code>stable&#95;sort&#95;by&#95;key</code> performs a key-value sort. That is, <code>stable&#95;sort&#95;by&#95;key</code> sorts the elements in <code>[keys&#95;first, keys&#95;last)</code> and <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> into ascending key order, meaning that if <code>i</code> and <code>j</code> are any two valid iterators in <code>[keys&#95;first, keys&#95;last)</code> such that <code>i</code> precedes <code>j</code>, and <code>p</code> and <code>q</code> are iterators in <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> corresponding to <code>i</code> and <code>j</code> respectively, then <code>&#42;j</code> is not less than <code>&#42;i</code>.

As the name suggests, <code>stable&#95;sort&#95;by&#95;key</code> is stable: it preserves the relative ordering of equivalent elements. That is, if <code>x</code> and <code>y</code> are elements in <code>[keys&#95;first, keys&#95;last)</code> such that <code>x</code> precedes <code>y</code>, and if the two elements are equivalent (neither <code>x &lt; y</code> nor <code>y &lt; x</code>) then a postcondition of <code>stable&#95;sort&#95;by&#95;key</code> is that <code>x</code> still precedes <code>y</code>.

This version of <code>stable&#95;sort&#95;by&#95;key</code> compares key objects using <code>operator&lt;</code>.


The following code snippet demonstrates how to use <code>stable&#95;sort&#95;by&#95;key</code> to sort an array of characters using integers as sorting keys.



```cpp
#include <thrust/sort.h>
...
const int N = 6;
int    keys[N] = {  1,   4,   2,   8,   5,   7};
char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};
thrust::stable_sort_by_key(keys, keys + N, values);
// keys is now   {  1,   2,   4,   5,   7,   8}
// values is now {'a', 'c', 'b', 'e', 'f', 'd'}
```

**Template Parameters**:
* **`RandomAccessIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>, <code>RandomAccessIterator1</code> is mutable, and <code>RandomAccessIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, and the ordering relation on <code>RandomAccessIterator1's</code><code>value&#95;type</code> is a _strict weak ordering_, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements. 
* **`RandomAccessIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator">Random Access Iterator</a>, and <code>RandomAccessIterator2</code> is mutable.

**Function Parameters**:
* **`keys_first`** The beginning of the key sequence. 
* **`keys_last`** The end of the key sequence. 
* **`values_first`** The beginning of the value sequence.

**Preconditions**:
The range <code>[keys&#95;first, keys&#95;last))</code> shall not overlap the range <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/sort">https://en.cppreference.com/w/cpp/algorithm/sort</a>
* <code>sort&#95;by&#95;key</code>
* <code>stable&#95;sort</code>

<h3 id="function-stable-sort-by-key">
Function <code>thrust::stable&#95;sort&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator1,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>__host__ __device__ void </span><span><b>stable_sort_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;RandomAccessIterator1 keys_first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator1 keys_last,</span>
<span>&nbsp;&nbsp;RandomAccessIterator2 values_first,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span></code>
<code>stable&#95;sort&#95;by&#95;key</code> performs a key-value sort. That is, <code>stable&#95;sort&#95;by&#95;key</code> sorts the elements in <code>[keys&#95;first, keys&#95;last)</code> and <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> into ascending key order, meaning that if <code>i</code> and <code>j</code> are any two valid iterators in <code>[keys&#95;first, keys&#95;last)</code> such that <code>i</code> precedes <code>j</code>, and <code>p</code> and <code>q</code> are iterators in <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> corresponding to <code>i</code> and <code>j</code> respectively, then <code>&#42;j</code> is not less than <code>&#42;i</code>.

As the name suggests, <code>stable&#95;sort&#95;by&#95;key</code> is stable: it preserves the relative ordering of equivalent elements. That is, if <code>x</code> and <code>y</code> are elements in <code>[keys&#95;first, keys&#95;last)</code> such that <code>x</code> precedes <code>y</code>, and if the two elements are equivalent (neither <code>x &lt; y</code> nor <code>y &lt; x</code>) then a postcondition of <code>stable&#95;sort&#95;by&#95;key</code> is that <code>x</code> still precedes <code>y</code>.

This version of <code>stable&#95;sort&#95;by&#95;key</code> compares key objects using the function object <code>comp</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>sort&#95;by&#95;key</code> to sort an array of character values using integers as sorting keys using the <code>thrust::host</code> execution policy for parallelization. The keys are sorted in descending order using the <code>greater&lt;int&gt;</code> comparison operator.



```cpp
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
...
const int N = 6;
int    keys[N] = {  1,   4,   2,   8,   5,   7};
char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};
thrust::stable_sort_by_key(thrust::host, keys, keys + N, values, thrust::greater<int>());
// keys is now   {  8,   7,   5,   4,   2,   1}
// values is now {'d', 'f', 'e', 'b', 'c', 'a'}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`RandomAccessIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>, <code>RandomAccessIterator1</code> is mutable, and <code>RandomAccessIterator1's</code><code>value&#95;type</code> is convertible to <code>StrictWeakOrdering's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>. 
* **`RandomAccessIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator">Random Access Iterator</a>, and <code>RandomAccessIterator2</code> is mutable. 
* **`StrictWeakOrdering`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`keys_first`** The beginning of the key sequence. 
* **`keys_last`** The end of the key sequence. 
* **`values_first`** The beginning of the value sequence. 
* **`comp`** Comparison operator.

**Preconditions**:
The range <code>[keys&#95;first, keys&#95;last))</code> shall not overlap the range <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/sort">https://en.cppreference.com/w/cpp/algorithm/sort</a>
* <code>sort&#95;by&#95;key</code>
* <code>stable&#95;sort</code>

<h3 id="function-stable-sort-by-key">
Function <code>thrust::stable&#95;sort&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename RandomAccessIterator1,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakOrdering&gt;</span>
<span>void </span><span><b>stable_sort_by_key</b>(RandomAccessIterator1 keys_first,</span>
<span>&nbsp;&nbsp;RandomAccessIterator1 keys_last,</span>
<span>&nbsp;&nbsp;RandomAccessIterator2 values_first,</span>
<span>&nbsp;&nbsp;StrictWeakOrdering comp);</span></code>
<code>stable&#95;sort&#95;by&#95;key</code> performs a key-value sort. That is, <code>stable&#95;sort&#95;by&#95;key</code> sorts the elements in <code>[keys&#95;first, keys&#95;last)</code> and <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> into ascending key order, meaning that if <code>i</code> and <code>j</code> are any two valid iterators in <code>[keys&#95;first, keys&#95;last)</code> such that <code>i</code> precedes <code>j</code>, and <code>p</code> and <code>q</code> are iterators in <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> corresponding to <code>i</code> and <code>j</code> respectively, then <code>&#42;j</code> is not less than <code>&#42;i</code>.

As the name suggests, <code>stable&#95;sort&#95;by&#95;key</code> is stable: it preserves the relative ordering of equivalent elements. That is, if <code>x</code> and <code>y</code> are elements in <code>[keys&#95;first, keys&#95;last)</code> such that <code>x</code> precedes <code>y</code>, and if the two elements are equivalent (neither <code>x &lt; y</code> nor <code>y &lt; x</code>) then a postcondition of <code>stable&#95;sort&#95;by&#95;key</code> is that <code>x</code> still precedes <code>y</code>.

This version of <code>stable&#95;sort&#95;by&#95;key</code> compares key objects using the function object <code>comp</code>.


The following code snippet demonstrates how to use <code>sort&#95;by&#95;key</code> to sort an array of character values using integers as sorting keys. The keys are sorted in descending order using the greater<int> comparison operator.



```cpp
#include <thrust/sort.h>
...
const int N = 6;
int    keys[N] = {  1,   4,   2,   8,   5,   7};
char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};
thrust::stable_sort_by_key(keys, keys + N, values, thrust::greater<int>());
// keys is now   {  8,   7,   5,   4,   2,   1}
// values is now {'d', 'f', 'e', 'b', 'c', 'a'}
```

**Template Parameters**:
* **`RandomAccessIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a>, <code>RandomAccessIterator1</code> is mutable, and <code>RandomAccessIterator1's</code><code>value&#95;type</code> is convertible to <code>StrictWeakOrdering's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>. 
* **`RandomAccessIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator">Random Access Iterator</a>, and <code>RandomAccessIterator2</code> is mutable. 
* **`StrictWeakOrdering`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`keys_first`** The beginning of the key sequence. 
* **`keys_last`** The end of the key sequence. 
* **`values_first`** The beginning of the value sequence. 
* **`comp`** Comparison operator.

**Preconditions**:
The range <code>[keys&#95;first, keys&#95;last))</code> shall not overlap the range <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/sort">https://en.cppreference.com/w/cpp/algorithm/sort</a>
* <code>sort&#95;by&#95;key</code>
* <code>stable&#95;sort</code>


