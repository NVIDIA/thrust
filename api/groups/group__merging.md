---
title: Merging
parent: Algorithms
grand_parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Merging

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__merging.html#function-merge">thrust::merge</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__merging.html#function-merge">thrust::merge</a></b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__merging.html#function-merge">thrust::merge</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__merging.html#function-merge">thrust::merge</a></b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__merging.html#function-merge-by-key">thrust::merge&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__merging.html#function-merge-by-key">thrust::merge&#95;by&#95;key</a></b>(InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename Compare&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__merging.html#function-merge-by-key">thrust::merge&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result,</span>
<span>&nbsp;&nbsp;Compare comp);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__merging.html#function-merge-by-key">thrust::merge&#95;by&#95;key</a></b>(InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span>
</code>

## Functions

<h3 id="function-merge">
Function <code>thrust::merge</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>merge</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>merge</code> combines two sorted ranges <code>[first1, last1)</code> and <code>[first2, last2)</code> into a single sorted range. That is, it copies from <code>[first1, last1)</code> and <code>[first2, last2)</code> into <code>[result, result + (last1 - first1) + (last2 - first2))</code> such that the resulting range is in ascending order. <code>merge</code> is stable, meaning both that the relative order of elements within each input range is preserved, and that for equivalent elements in both input ranges the element from the first range precedes the element from the second. The return value is <code>result + (last1 - first1) + (last2 - first2)</code>.

This version of <code>merge</code> compares elements using <code>operator&lt;</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>merge</code> to compute the merger of two sorted sets of integers using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/merge.h>
#include <thrust/execution_policy.h>
...
int A1[6] = {1, 3, 5, 7, 9, 11};
int A2[7] = {1, 1, 2, 3, 5,  8, 13};

int result[13];

int *result_end =
  thrust::merge(thrust::host,
                A1, A1 + 6,
                A2, A2 + 7,
                result);
// result = {1, 1, 1, 2, 3, 3, 5, 5, 7, 8, 9, 11, 13}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the first input range. 
* **`last1`** The end of the first input range. 
* **`first2`** The beginning of the second input range. 
* **`last2`** The end of the second input range. 
* **`result`** The beginning of the merged output. 

**Preconditions**:
* The ranges <code>[first1, last1)</code> and <code>[first2, last2)</code> shall be sorted with respect to <code>operator&lt;</code>. 
* The resulting range shall not overlap with either input range.

**Returns**:
The end of the output range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/merge">https://en.cppreference.com/w/cpp/algorithm/merge</a>
* <code>set&#95;union</code>
* <code>sort</code>
* <code>is&#95;sorted</code>

<h3 id="function-merge">
Function <code>thrust::merge</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b>merge</b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>merge</code> combines two sorted ranges <code>[first1, last1)</code> and <code>[first2, last2)</code> into a single sorted range. That is, it copies from <code>[first1, last1)</code> and <code>[first2, last2)</code> into <code>[result, result + (last1 - first1) + (last2 - first2))</code> such that the resulting range is in ascending order. <code>merge</code> is stable, meaning both that the relative order of elements within each input range is preserved, and that for equivalent elements in both input ranges the element from the first range precedes the element from the second. The return value is <code>result + (last1 - first1) + (last2 - first2)</code>.

This version of <code>merge</code> compares elements using <code>operator&lt;</code>.


The following code snippet demonstrates how to use <code>merge</code> to compute the merger of two sorted sets of integers.



```cpp
#include <thrust/merge.h>
...
int A1[6] = {1, 3, 5, 7, 9, 11};
int A2[7] = {1, 1, 2, 3, 5,  8, 13};

int result[13];

int *result_end = thrust::merge(A1, A1 + 6, A2, A2 + 7, result);
// result = {1, 1, 1, 2, 3, 3, 5, 5, 7, 8, 9, 11, 13}
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`first1`** The beginning of the first input range. 
* **`last1`** The end of the first input range. 
* **`first2`** The beginning of the second input range. 
* **`last2`** The end of the second input range. 
* **`result`** The beginning of the merged output. 

**Preconditions**:
* The ranges <code>[first1, last1)</code> and <code>[first2, last2)</code> shall be sorted with respect to <code>operator&lt;</code>. 
* The resulting range shall not overlap with either input range.

**Returns**:
The end of the output range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/merge">https://en.cppreference.com/w/cpp/algorithm/merge</a>
* <code>set&#95;union</code>
* <code>sort</code>
* <code>is&#95;sorted</code>

<h3 id="function-merge">
Function <code>thrust::merge</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>merge</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span></code>
<code>merge</code> combines two sorted ranges <code>[first1, last1)</code> and <code>[first2, last2)</code> into a single sorted range. That is, it copies from <code>[first1, last1)</code> and <code>[first2, last2)</code> into <code>[result, result + (last1 - first1) + (last2 - first2))</code> such that the resulting range is in ascending order. <code>merge</code> is stable, meaning both that the relative order of elements within each input range is preserved, and that for equivalent elements in both input ranges the element from the first range precedes the element from the second. The return value is <code>result + (last1 - first1) + (last2 - first2)</code>.

This version of <code>merge</code> compares elements using a function object <code>comp</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>merge</code> to compute the merger of two sets of integers sorted in descending order using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/merge.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...
int A1[6] = {11, 9, 7, 5, 3, 1};
int A2[7] = {13, 8, 5, 3, 2, 1, 1};

int result[13];

int *result_end = thrust::merge(thrust::host,
                                A1, A1 + 6,
                                A2, A2 + 7,
                                result,
                                thrust::greater<int>());
// result = {13, 11, 9, 8, 7, 5, 5, 3, 3, 2, 1, 1, 1}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1's</code><code>value&#95;type</code> is convertable to <code>StrictWeakCompare's</code><code>first&#95;argument&#95;type</code>. and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2's</code><code>value&#95;type</code> is convertable to <code>StrictWeakCompare's</code><code>second&#95;argument&#95;type</code>. and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`StrictWeakCompare`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the first input range. 
* **`last1`** The end of the first input range. 
* **`first2`** The beginning of the second input range. 
* **`last2`** The end of the second input range. 
* **`result`** The beginning of the merged output. 
* **`comp`** Comparison operator. 

**Preconditions**:
* The ranges <code>[first1, last1)</code> and <code>[first2, last2)</code> shall be sorted with respect to <code>comp</code>. 
* The resulting range shall not overlap with either input range.

**Returns**:
The end of the output range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/merge">https://en.cppreference.com/w/cpp/algorithm/merge</a>
* <code>sort</code>
* <code>is&#95;sorted</code>

<h3 id="function-merge">
Function <code>thrust::merge</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span>OutputIterator </span><span><b>merge</b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;InputIterator2 last2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span></code>
<code>merge</code> combines two sorted ranges <code>[first1, last1)</code> and <code>[first2, last2)</code> into a single sorted range. That is, it copies from <code>[first1, last1)</code> and <code>[first2, last2)</code> into <code>[result, result + (last1 - first1) + (last2 - first2))</code> such that the resulting range is in ascending order. <code>merge</code> is stable, meaning both that the relative order of elements within each input range is preserved, and that for equivalent elements in both input ranges the element from the first range precedes the element from the second. The return value is <code>result + (last1 - first1) + (last2 - first2)</code>.

This version of <code>merge</code> compares elements using a function object <code>comp</code>.


The following code snippet demonstrates how to use <code>merge</code> to compute the merger of two sets of integers sorted in descending order.



```cpp
#include <thrust/merge.h>
#include <thrust/functional.h>
...
int A1[6] = {11, 9, 7, 5, 3, 1};
int A2[7] = {13, 8, 5, 3, 2, 1, 1};

int result[13];

int *result_end = thrust::merge(A1, A1 + 6, A2, A2 + 7, result, thrust::greater<int>());
// result = {13, 11, 9, 8, 7, 5, 5, 3, 3, 2, 1, 1, 1}
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1's</code><code>value&#95;type</code> is convertable to <code>StrictWeakCompare's</code><code>first&#95;argument&#95;type</code>. and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2's</code><code>value&#95;type</code> is convertable to <code>StrictWeakCompare's</code><code>second&#95;argument&#95;type</code>. and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`StrictWeakCompare`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`first1`** The beginning of the first input range. 
* **`last1`** The end of the first input range. 
* **`first2`** The beginning of the second input range. 
* **`last2`** The end of the second input range. 
* **`result`** The beginning of the merged output. 
* **`comp`** Comparison operator. 

**Preconditions**:
* The ranges <code>[first1, last1)</code> and <code>[first2, last2)</code> shall be sorted with respect to <code>comp</code>. 
* The resulting range shall not overlap with either input range.

**Returns**:
The end of the output range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/merge">https://en.cppreference.com/w/cpp/algorithm/merge</a>
* <code>sort</code>
* <code>is&#95;sorted</code>

<h3 id="function-merge-by-key">
Function <code>thrust::merge&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>merge_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result);</span></code>
<code>merge&#95;by&#95;key</code> performs a key-value merge. That is, <code>merge&#95;by&#95;key</code> copies elements from <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code> into a single range, <code>[keys&#95;result, keys&#95;result + (keys&#95;last1 - keys&#95;first1) + (keys&#95;last2 - keys&#95;first2))</code> such that the resulting range is in ascending key order.

At the same time, <code>merge&#95;by&#95;key</code> copies elements from the two associated ranges <code>[values&#95;first1 + (keys&#95;last1 - keys&#95;first1))</code> and <code>[values&#95;first2 + (keys&#95;last2 - keys&#95;first2))</code> into a single range, <code>[values&#95;result, values&#95;result + (keys&#95;last1 - keys&#95;first1) + (keys&#95;last2 - keys&#95;first2))</code> such that the resulting range is in ascending order implied by each input element's associated key.

<code>merge&#95;by&#95;key</code> is stable, meaning both that the relative order of elements within each input range is preserved, and that for equivalent elements in all input key ranges the element from the first range precedes the element from the second.

The return value is is <code>(keys&#95;result + (keys&#95;last1 - keys&#95;first1) + (keys&#95;last2 - keys&#95;first2))</code> and <code>(values&#95;result + (keys&#95;last1 - keys&#95;first1) + (keys&#95;last2 - keys&#95;first2))</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>merge&#95;by&#95;key</code> to compute the merger of two sets of integers sorted in ascending order using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/merge.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...
int A_keys[6] = {1, 3, 5, 7, 9, 11};
int A_vals[6] = {0, 0, 0, 0, 0, 0};

int B_keys[7] = {1, 1, 2, 3, 5, 8, 13};
int B_vals[7] = {1, 1, 1, 1, 1, 1, 1};

int keys_result[13];
int vals_result[13];

thrust::pair<int*,int*> end =
  thrust::merge_by_key(thrust::host,
                       A_keys, A_keys + 6,
                       B_keys, B_keys + 7,
                       A_vals, B_vals,
                       keys_result, vals_result);

// keys_result = {1, 1, 1, 2, 3, 3, 5, 5, 7, 8, 9, 11, 13}
// vals_result = {0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0,  0,  1}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator3`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator3's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`InputIterator4`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator4's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`keys_first1`** The beginning of the first input range of keys. 
* **`keys_last1`** The end of the first input range of keys. 
* **`keys_first2`** The beginning of the second input range of keys. 
* **`keys_last2`** The end of the second input range of keys. 
* **`values_first1`** The beginning of the first input range of values. 
* **`values_first2`** The beginning of the first input range of values. 
* **`keys_result`** The beginning of the merged output range of keys. 
* **`values_result`** The beginning of the merged output range of values. 

**Preconditions**:
* The ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code> shall be sorted with respect to <code>operator&lt;</code>. 
* The resulting ranges shall not overlap with any input range.

**Returns**:
A <code>pair</code><code>p</code> such that <code>p.first</code> is the end of the output range of keys, and such that <code>p.second</code> is the end of the output range of values.

**See**:
* merge 
* <code>sort&#95;by&#95;key</code>
* <code>is&#95;sorted</code>

<h3 id="function-merge-by-key">
Function <code>thrust::merge&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>merge_by_key</b>(InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result);</span></code>
<code>merge&#95;by&#95;key</code> performs a key-value merge. That is, <code>merge&#95;by&#95;key</code> copies elements from <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code> into a single range, <code>[keys&#95;result, keys&#95;result + (keys&#95;last1 - keys&#95;first1) + (keys&#95;last2 - keys&#95;first2))</code> such that the resulting range is in ascending key order.

At the same time, <code>merge&#95;by&#95;key</code> copies elements from the two associated ranges <code>[values&#95;first1 + (keys&#95;last1 - keys&#95;first1))</code> and <code>[values&#95;first2 + (keys&#95;last2 - keys&#95;first2))</code> into a single range, <code>[values&#95;result, values&#95;result + (keys&#95;last1 - keys&#95;first1) + (keys&#95;last2 - keys&#95;first2))</code> such that the resulting range is in ascending order implied by each input element's associated key.

<code>merge&#95;by&#95;key</code> is stable, meaning both that the relative order of elements within each input range is preserved, and that for equivalent elements in all input key ranges the element from the first range precedes the element from the second.

The return value is is <code>(keys&#95;result + (keys&#95;last1 - keys&#95;first1) + (keys&#95;last2 - keys&#95;first2))</code> and <code>(values&#95;result + (keys&#95;last1 - keys&#95;first1) + (keys&#95;last2 - keys&#95;first2))</code>.


The following code snippet demonstrates how to use <code>merge&#95;by&#95;key</code> to compute the merger of two sets of integers sorted in ascending order.



```cpp
#include <thrust/merge.h>
#include <thrust/functional.h>
...
int A_keys[6] = {1, 3, 5, 7, 9, 11};
int A_vals[6] = {0, 0, 0, 0, 0, 0};

int B_keys[7] = {1, 1, 2, 3, 5, 8, 13};
int B_vals[7] = {1, 1, 1, 1, 1, 1, 1};

int keys_result[13];
int vals_result[13];

thrust::pair<int*,int*> end = thrust::merge_by_key(A_keys, A_keys + 6, B_keys, B_keys + 7, A_vals, B_vals, keys_result, vals_result);

// keys_result = {1, 1, 1, 2, 3, 3, 5, 5, 7, 8, 9, 11, 13}
// vals_result = {0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0,  0,  1}
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1</code> and <code>InputIterator2</code> have the same <code>value&#95;type</code>, <code>InputIterator1's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator1's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2</code> and <code>InputIterator1</code> have the same <code>value&#95;type</code>, <code>InputIterator2's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>, the ordering on <code>InputIterator2's</code><code>value&#95;type</code> is a strict weak ordering, as defined in the <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a> requirements, and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator3`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator3's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`InputIterator4`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator4's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`keys_first1`** The beginning of the first input range of keys. 
* **`keys_last1`** The end of the first input range of keys. 
* **`keys_first2`** The beginning of the second input range of keys. 
* **`keys_last2`** The end of the second input range of keys. 
* **`values_first1`** The beginning of the first input range of values. 
* **`values_first2`** The beginning of the first input range of values. 
* **`keys_result`** The beginning of the merged output range of keys. 
* **`values_result`** The beginning of the merged output range of values. 

**Preconditions**:
* The ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code> shall be sorted with respect to <code>operator&lt;</code>. 
* The resulting ranges shall not overlap with any input range.

**Returns**:
A <code>pair</code><code>p</code> such that <code>p.first</code> is the end of the output range of keys, and such that <code>p.second</code> is the end of the output range of values.

**See**:
* merge 
* <code>sort&#95;by&#95;key</code>
* <code>is&#95;sorted</code>

<h3 id="function-merge-by-key">
Function <code>thrust::merge&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename Compare&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>merge_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result,</span>
<span>&nbsp;&nbsp;Compare comp);</span></code>
<code>merge&#95;by&#95;key</code> performs a key-value merge. That is, <code>merge&#95;by&#95;key</code> copies elements from <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code> into a single range, <code>[keys&#95;result, keys&#95;result + (keys&#95;last1 - keys&#95;first1) + (keys&#95;last2 - keys&#95;first2))</code> such that the resulting range is in ascending key order.

At the same time, <code>merge&#95;by&#95;key</code> copies elements from the two associated ranges <code>[values&#95;first1 + (keys&#95;last1 - keys&#95;first1))</code> and <code>[values&#95;first2 + (keys&#95;last2 - keys&#95;first2))</code> into a single range, <code>[values&#95;result, values&#95;result + (keys&#95;last1 - keys&#95;first1) + (keys&#95;last2 - keys&#95;first2))</code> such that the resulting range is in ascending order implied by each input element's associated key.

<code>merge&#95;by&#95;key</code> is stable, meaning both that the relative order of elements within each input range is preserved, and that for equivalent elements in all input key ranges the element from the first range precedes the element from the second.

The return value is is <code>(keys&#95;result + (keys&#95;last1 - keys&#95;first1) + (keys&#95;last2 - keys&#95;first2))</code> and <code>(values&#95;result + (keys&#95;last1 - keys&#95;first1) + (keys&#95;last2 - keys&#95;first2))</code>.

This version of <code>merge&#95;by&#95;key</code> compares key elements using a function object <code>comp</code>.

The algorithm's execution is parallelized using <code>exec</code>.


The following code snippet demonstrates how to use <code>merge&#95;by&#95;key</code> to compute the merger of two sets of integers sorted in descending order using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/merge.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...
int A_keys[6] = {11, 9, 7, 5, 3, 1};
int A_vals[6] = { 0, 0, 0, 0, 0, 0};

int B_keys[7] = {13, 8, 5, 3, 2, 1, 1};
int B_vals[7] = { 1, 1, 1, 1, 1, 1, 1};

int keys_result[13];
int vals_result[13];

thrust::pair<int*,int*> end =
  thrust::merge_by_key(thrust::host,
                       A_keys, A_keys + 6,
                       B_keys, B_keys + 7,
                       A_vals, B_vals,
                       keys_result, vals_result,
                       thrust::greater<int>());

// keys_result = {13, 11, 9, 8, 7, 5, 5, 3, 3, 2, 1, 1, 1}
// vals_result = { 1,  0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1's</code><code>value&#95;type</code> is convertable to <code>StrictWeakCompare's</code><code>first&#95;argument&#95;type</code>. and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator1's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2's</code><code>value&#95;type</code> is convertable to <code>StrictWeakCompare's</code><code>second&#95;argument&#95;type</code>. and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator1's</code> set of <code>value&#95;types</code>. 
* **`InputIterator3`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator3's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`InputIterator4`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator4's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`StrictWeakCompare`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`keys_first1`** The beginning of the first input range of keys. 
* **`keys_last1`** The end of the first input range of keys. 
* **`keys_first2`** The beginning of the second input range of keys. 
* **`keys_last2`** The end of the second input range of keys. 
* **`values_first1`** The beginning of the first input range of values. 
* **`values_first2`** The beginning of the first input range of values. 
* **`keys_result`** The beginning of the merged output range of keys. 
* **`values_result`** The beginning of the merged output range of values. 
* **`comp`** Comparison operator. 

**Preconditions**:
* The ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code> shall be sorted with respect to <code>comp</code>. 
* The resulting ranges shall not overlap with any input range.

**Returns**:
A <code>pair</code><code>p</code> such that <code>p.first</code> is the end of the output range of keys, and such that <code>p.second</code> is the end of the output range of values.

**See**:
* merge 
* <code>sort&#95;by&#95;key</code>
* <code>is&#95;sorted</code>

<h3 id="function-merge-by-key">
Function <code>thrust::merge&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename InputIterator4,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename StrictWeakCompare&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>merge_by_key</b>(InputIterator1 keys_first1,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last1,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_first2,</span>
<span>&nbsp;&nbsp;InputIterator2 keys_last2,</span>
<span>&nbsp;&nbsp;InputIterator3 values_first1,</span>
<span>&nbsp;&nbsp;InputIterator4 values_first2,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result,</span>
<span>&nbsp;&nbsp;StrictWeakCompare comp);</span></code>
<code>merge&#95;by&#95;key</code> performs a key-value merge. That is, <code>merge&#95;by&#95;key</code> copies elements from <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code> into a single range, <code>[keys&#95;result, keys&#95;result + (keys&#95;last1 - keys&#95;first1) + (keys&#95;last2 - keys&#95;first2))</code> such that the resulting range is in ascending key order.

At the same time, <code>merge&#95;by&#95;key</code> copies elements from the two associated ranges <code>[values&#95;first1 + (keys&#95;last1 - keys&#95;first1))</code> and <code>[values&#95;first2 + (keys&#95;last2 - keys&#95;first2))</code> into a single range, <code>[values&#95;result, values&#95;result + (keys&#95;last1 - keys&#95;first1) + (keys&#95;last2 - keys&#95;first2))</code> such that the resulting range is in ascending order implied by each input element's associated key.

<code>merge&#95;by&#95;key</code> is stable, meaning both that the relative order of elements within each input range is preserved, and that for equivalent elements in all input key ranges the element from the first range precedes the element from the second.

The return value is is <code>(keys&#95;result + (keys&#95;last1 - keys&#95;first1) + (keys&#95;last2 - keys&#95;first2))</code> and <code>(values&#95;result + (keys&#95;last1 - keys&#95;first1) + (keys&#95;last2 - keys&#95;first2))</code>.

This version of <code>merge&#95;by&#95;key</code> compares key elements using a function object <code>comp</code>.


The following code snippet demonstrates how to use <code>merge&#95;by&#95;key</code> to compute the merger of two sets of integers sorted in descending order.



```cpp
#include <thrust/merge.h>
#include <thrust/functional.h>
...
int A_keys[6] = {11, 9, 7, 5, 3, 1};
int A_vals[6] = { 0, 0, 0, 0, 0, 0};

int B_keys[7] = {13, 8, 5, 3, 2, 1, 1};
int B_vals[7] = { 1, 1, 1, 1, 1, 1, 1};

int keys_result[13];
int vals_result[13];

thrust::pair<int*,int*> end = thrust::merge_by_key(A_keys, A_keys + 6, B_keys, B_keys + 7, A_vals, B_vals, keys_result, vals_result, thrust::greater<int>());

// keys_result = {13, 11, 9, 8, 7, 5, 5, 3, 3, 2, 1, 1, 1}
// vals_result = { 1,  0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1}
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1's</code><code>value&#95;type</code> is convertable to <code>StrictWeakCompare's</code><code>first&#95;argument&#95;type</code>. and <code>InputIterator1's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator1's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator2's</code><code>value&#95;type</code> is convertable to <code>StrictWeakCompare's</code><code>second&#95;argument&#95;type</code>. and <code>InputIterator2's</code><code>value&#95;type</code> is convertable to a type in <code>OutputIterator1's</code> set of <code>value&#95;types</code>. 
* **`InputIterator3`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator3's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`InputIterator4`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator4's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator2's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`StrictWeakCompare`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/strict_weak_order">Strict Weak Ordering</a>.

**Function Parameters**:
* **`keys_first1`** The beginning of the first input range of keys. 
* **`keys_last1`** The end of the first input range of keys. 
* **`keys_first2`** The beginning of the second input range of keys. 
* **`keys_last2`** The end of the second input range of keys. 
* **`values_first1`** The beginning of the first input range of values. 
* **`values_first2`** The beginning of the first input range of values. 
* **`keys_result`** The beginning of the merged output range of keys. 
* **`values_result`** The beginning of the merged output range of values. 
* **`comp`** Comparison operator. 

**Preconditions**:
* The ranges <code>[keys&#95;first1, keys&#95;last1)</code> and <code>[keys&#95;first2, keys&#95;last2)</code> shall be sorted with respect to <code>comp</code>. 
* The resulting ranges shall not overlap with any input range.

**Returns**:
A <code>pair</code><code>p</code> such that <code>p.first</code> is the end of the output range of keys, and such that <code>p.second</code> is the end of the output range of values.

**See**:
* merge 
* <code>sort&#95;by&#95;key</code>
* <code>is&#95;sorted</code>


