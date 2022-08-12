---
title: Stream Compaction
parent: Reordering
grand_parent: Algorithms
nav_exclude: false
has_children: true
has_toc: false
---

# Stream Compaction

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-copy-if">thrust::copy&#95;if</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-copy-if">thrust::copy&#95;if</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-copy-if">thrust::copy&#95;if</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-copy-if">thrust::copy&#95;if</a></b>(InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-remove">thrust::remove</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & value);</span>
<br>
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-remove">thrust::remove</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & value);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-remove-copy">thrust::remove&#95;copy</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;const T & value);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-remove-copy">thrust::remove&#95;copy</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;const T & value);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-remove-if">thrust::remove&#95;if</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-remove-if">thrust::remove&#95;if</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-remove-copy-if">thrust::remove&#95;copy&#95;if</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-remove-copy-if">thrust::remove&#95;copy&#95;if</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-remove-if">thrust::remove&#95;if</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator stencil,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-remove-if">thrust::remove&#95;if</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator stencil,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-remove-copy-if">thrust::remove&#95;copy&#95;if</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-remove-copy-if">thrust::remove&#95;copy&#95;if</a></b>(InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-unique">thrust::unique</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span>
<br>
<span>template &lt;typename ForwardIterator&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-unique">thrust::unique</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-unique">thrust::unique</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span>
<br>
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-unique">thrust::unique</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-unique-copy">thrust::unique&#95;copy</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-unique-copy">thrust::unique&#95;copy</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-unique-copy">thrust::unique&#95;copy</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-unique-copy">thrust::unique&#95;copy</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator1,</span>
<span>&nbsp;&nbsp;typename ForwardIterator2&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< ForwardIterator1, ForwardIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-unique-by-key">thrust::unique&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator1 keys_first,</span>
<span>&nbsp;&nbsp;ForwardIterator1 keys_last,</span>
<span>&nbsp;&nbsp;ForwardIterator2 values_first);</span>
<br>
<span>template &lt;typename ForwardIterator1,</span>
<span>&nbsp;&nbsp;typename ForwardIterator2&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< ForwardIterator1, ForwardIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-unique-by-key">thrust::unique&#95;by&#95;key</a></b>(ForwardIterator1 keys_first,</span>
<span>&nbsp;&nbsp;ForwardIterator1 keys_last,</span>
<span>&nbsp;&nbsp;ForwardIterator2 values_first);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator1,</span>
<span>&nbsp;&nbsp;typename ForwardIterator2,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< ForwardIterator1, ForwardIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-unique-by-key">thrust::unique&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator1 keys_first,</span>
<span>&nbsp;&nbsp;ForwardIterator1 keys_last,</span>
<span>&nbsp;&nbsp;ForwardIterator2 values_first,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span>
<br>
<span>template &lt;typename ForwardIterator1,</span>
<span>&nbsp;&nbsp;typename ForwardIterator2,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< ForwardIterator1, ForwardIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-unique-by-key">thrust::unique&#95;by&#95;key</a></b>(ForwardIterator1 keys_first,</span>
<span>&nbsp;&nbsp;ForwardIterator1 keys_last,</span>
<span>&nbsp;&nbsp;ForwardIterator2 values_first,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-unique-by-key-copy">thrust::unique&#95;by&#95;key&#95;copy</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last,</span>
<span>&nbsp;&nbsp;InputIterator2 values_first,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-unique-by-key-copy">thrust::unique&#95;by&#95;key&#95;copy</a></b>(InputIterator1 keys_first,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last,</span>
<span>&nbsp;&nbsp;InputIterator2 values_first,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-unique-by-key-copy">thrust::unique&#95;by&#95;key&#95;copy</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last,</span>
<span>&nbsp;&nbsp;InputIterator2 values_first,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-unique-by-key-copy">thrust::unique&#95;by&#95;key&#95;copy</a></b>(InputIterator1 keys_first,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last,</span>
<span>&nbsp;&nbsp;InputIterator2 values_first,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">thrust::iterator_traits</a>< ForwardIterator >::difference_type </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-unique-count">thrust::unique&#95;count</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">thrust::iterator_traits</a>< ForwardIterator >::difference_type </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-unique-count">thrust::unique&#95;count</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span>
<br>
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">thrust::iterator_traits</a>< ForwardIterator >::difference_type </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-unique-count">thrust::unique&#95;count</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span>
<br>
<span>template &lt;typename ForwardIterator&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">thrust::iterator_traits</a>< ForwardIterator >::difference_type </span><span><b><a href="{{ site.baseurl }}/api/groups/group__stream__compaction.html#function-unique-count">thrust::unique&#95;count</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span>
</code>

## Functions

<h3 id="function-copy-if">
Function <code>thrust::copy&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>copy_if</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
This version of <code>copy&#95;if</code> copies elements from the range <code>[first,last)</code> to a range beginning at <code>result</code>, except that any element which causes <code>pred</code> to be <code>false</code> is not copied. <code>copy&#95;if</code> is stable, meaning that the relative order of elements that are copied is unchanged.

More precisely, for every integer <code>n</code> such that <code>0 &lt;= n &lt; last-first</code>, <code>copy&#95;if</code> performs the assignment <code>&#42;result = &#42;(first+n)</code> and <code>result</code> is advanced one position if <code>pred(&#42;(first+n))</code>. Otherwise, no assignment occurs and <code>result</code> is not advanced.

The algorithm's execution is parallelized as determined by <code>system</code>.


The following code snippet demonstrates how to use <code>copy&#95;if</code> to perform stream compaction to copy even numbers to an output range using the <code>thrust::host</code> parallelization policy:



```cpp
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
...
struct is_even
{
  __host__ __device__
  bool operator()(const int x)
  {
    return (x % 2) == 0;
  }
};
...
const int N = 6;
int V[N] = {-2, 0, -1, 0, 1, 2};
int result[4];

thrust::copy_if(thrust::host, V, V + N, result, is_even());

// V remains {-2, 0, -1, 0, 1, 2}
// result is now {-2, 0, 0, 2}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence from which to copy. 
* **`last`** The end of the sequence from which to copy. 
* **`result`** The beginning of the sequence into which to copy. 
* **`pred`** The predicate to test on every value of the range <code>[first, last)</code>. 

**Preconditions**:
The ranges <code>[first, last)</code> and <code>[result, result + (last - first))</code> shall not overlap.

**Returns**:
<code>result + n</code>, where <code>n</code> is equal to the number of times <code>pred</code> evaluated to <code>true</code> in the range <code>[first, last)</code>.

**See**:
<code>remove&#95;copy&#95;if</code>

<h3 id="function-copy-if">
Function <code>thrust::copy&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>OutputIterator </span><span><b>copy_if</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
This version of <code>copy&#95;if</code> copies elements from the range <code>[first,last)</code> to a range beginning at <code>result</code>, except that any element which causes <code>pred</code> to <code>false</code> is not copied. <code>copy&#95;if</code> is stable, meaning that the relative order of elements that are copied is unchanged.

More precisely, for every integer <code>n</code> such that <code>0 &lt;= n &lt; last-first</code>, <code>copy&#95;if</code> performs the assignment <code>&#42;result = &#42;(first+n)</code> and <code>result</code> is advanced one position if <code>pred(&#42;(first+n))</code>. Otherwise, no assignment occurs and <code>result</code> is not advanced.


The following code snippet demonstrates how to use <code>copy&#95;if</code> to perform stream compaction to copy even numbers to an output range.



```cpp
#include <thrust/copy.h>
...
struct is_even
{
  __host__ __device__
  bool operator()(const int x)
  {
    return (x % 2) == 0;
  }
};
...
const int N = 6;
int V[N] = {-2, 0, -1, 0, 1, 2};
int result[4];

thrust::copy_if(V, V + N, result, is_even());

// V remains {-2, 0, -1, 0, 1, 2}
// result is now {-2, 0, 0, 2}
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first`** The beginning of the sequence from which to copy. 
* **`last`** The end of the sequence from which to copy. 
* **`result`** The beginning of the sequence into which to copy. 
* **`pred`** The predicate to test on every value of the range <code>[first, last)</code>. 

**Preconditions**:
The ranges <code>[first, last)</code> and <code>[result, result + (last - first))</code> shall not overlap.

**Returns**:
<code>result + n</code>, where <code>n</code> is equal to the number of times <code>pred</code> evaluated to <code>true</code> in the range <code>[first, last)</code>.

**See**:
<code>remove&#95;copy&#95;if</code>

<h3 id="function-copy-if">
Function <code>thrust::copy&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>copy_if</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
This version of <code>copy&#95;if</code> copies elements from the range <code>[first,last)</code> to a range beginning at <code>result</code>, except that any element whose corresponding stencil element causes <code>pred</code> to be <code>false</code> is not copied. <code>copy&#95;if</code> is stable, meaning that the relative order of elements that are copied is unchanged.

More precisely, for every integer <code>n</code> such that <code>0 &lt;= n &lt; last-first</code>, <code>copy&#95;if</code> performs the assignment <code>&#42;result = &#42;(first+n)</code> and <code>result</code> is advanced one position if <code>pred(&#42;(stencil+n))</code>. Otherwise, no assignment occurs and <code>result</code> is not advanced.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>copy&#95;if</code> to perform stream compaction to copy numbers to an output range when corresponding stencil elements are even using the <code>thrust::host</code> execution policy:



```cpp
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
...
struct is_even
{
  __host__ __device__
  bool operator()(const int x)
  {
    return (x % 2) == 0;
  }
};
...
int N = 6;
int data[N]    = { 0, 1,  2, 3, 4, 5};
int stencil[N] = {-2, 0, -1, 0, 1, 2};
int result[4];

thrust::copy_if(thrust::host, data, data + N, stencil, result, is_even());

// data remains    = { 0, 1,  2, 3, 4, 5};
// stencil remains = {-2, 0, -1, 0, 1, 2};
// result is now     { 0, 1,  3, 5}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/OutputIterator">Output Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence from which to copy. 
* **`last`** The end of the sequence from which to copy. 
* **`stencil`** The beginning of the stencil sequence. 
* **`result`** The beginning of the sequence into which to copy. 
* **`pred`** The predicate to test on every value of the range <code>[stencil, stencil + (last-first))</code>. 

**Preconditions**:
* The ranges <code>[first, last)</code> and <code>[result, result + (last - first))</code> shall not overlap. 
* The ranges <code>[stencil, stencil + (last - first))</code> and <code>[result, result + (last - first))</code> shall not overlap.

**Returns**:
<code>result + n</code>, where <code>n</code> is equal to the number of times <code>pred</code> evaluated to <code>true</code> in the range <code>[stencil, stencil + (last-first))</code>.

**See**:
<code>remove&#95;copy&#95;if</code>

<h3 id="function-copy-if">
Function <code>thrust::copy&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>OutputIterator </span><span><b>copy_if</b>(InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
This version of <code>copy&#95;if</code> copies elements from the range <code>[first,last)</code> to a range beginning at <code>result</code>, except that any element whose corresponding stencil element causes <code>pred</code> to be <code>false</code> is not copied. <code>copy&#95;if</code> is stable, meaning that the relative order of elements that are copied is unchanged.

More precisely, for every integer <code>n</code> such that <code>0 &lt;= n &lt; last-first</code>, <code>copy&#95;if</code> performs the assignment <code>&#42;result = &#42;(first+n)</code> and <code>result</code> is advanced one position if <code>pred(&#42;(stencil+n))</code>. Otherwise, no assignment occurs and <code>result</code> is not advanced.


The following code snippet demonstrates how to use <code>copy&#95;if</code> to perform stream compaction to copy numbers to an output range when corresponding stencil elements are even:



```cpp
#include <thrust/copy.h>
...
struct is_even
{
  __host__ __device__
  bool operator()(const int x)
  {
    return (x % 2) == 0;
  }
};
...
int N = 6;
int data[N]    = { 0, 1,  2, 3, 4, 5};
int stencil[N] = {-2, 0, -1, 0, 1, 2};
int result[4];

thrust::copy_if(data, data + N, stencil, result, is_even());

// data remains    = { 0, 1,  2, 3, 4, 5};
// stencil remains = {-2, 0, -1, 0, 1, 2};
// result is now     { 0, 1,  3, 5}
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/OutputIterator">Output Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first`** The beginning of the sequence from which to copy. 
* **`last`** The end of the sequence from which to copy. 
* **`stencil`** The beginning of the stencil sequence. 
* **`result`** The beginning of the sequence into which to copy. 
* **`pred`** The predicate to test on every value of the range <code>[stencil, stencil + (last-first))</code>. 

**Preconditions**:
* The ranges <code>[first, last)</code> and <code>[result, result + (last - first))</code> shall not overlap. 
* The ranges <code>[stencil, stencil + (last - first))</code> and <code>[result, result + (last - first))</code> shall not overlap.

**Returns**:
<code>result + n</code>, where <code>n</code> is equal to the number of times <code>pred</code> evaluated to <code>true</code> in the range <code>[stencil, stencil + (last-first))</code>.

**See**:
<code>remove&#95;copy&#95;if</code>

<h3 id="function-remove">
Function <code>thrust::remove</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>remove</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & value);</span></code>
<code>remove</code> removes from the range <code>[first, last)</code> all elements that are equal to <code>value</code>. That is, <code>remove</code> returns an iterator <code>new&#95;last</code> such that the range <code>[first, new&#95;last)</code> contains no elements equal to <code>value</code>. The iterators in the range <code>[new&#95;first,last)</code> are all still dereferenceable, but the elements that they point to are unspecified. <code>remove</code> is stable, meaning that the relative order of elements that are not equal to <code>value</code> is unchanged.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>remove</code> to remove a number of interest from a range using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
...
const int N = 6;
int A[N] = {3, 1, 4, 1, 5, 9};
int *new_end = thrust::remove(A, A + N, 1);
// The first four values of A are now {3, 4, 5, 9}
// Values beyond new_end are unspecified
```

**Note**:
The meaning of "removal" is somewhat subtle. <code>remove</code> does not destroy any iterators, and does not change the distance between <code>first</code> and <code>last</code>. (There's no way that it could do anything of the sort.) So, for example, if <code>V</code> is a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device_vector</a>, <code>remove(V.begin(), V.end(), 0)</code> does not change <code>V.size()</code>: <code>V</code> will contain just as many elements as it did before. <code>remove</code> returns an iterator that points to the end of the resulting range after elements have been removed from it; it follows that the elements after that iterator are of no interest, and may be discarded. If you are removing elements from a <a href="https://en.cppreference.com/w/cpp/container">Sequence</a>, you may simply erase them. That is, a reasonable way of removing elements from a <a href="https://en.cppreference.com/w/cpp/container">Sequence</a> is <code>S.erase(remove(S.begin(), S.end(), x), S.end())</code>.

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>, and objects of type <code>T</code> can be compared for equality with objects of <code>ForwardIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the range of interest. 
* **`last`** The end of the range of interest. 
* **`value`** The value to remove from the range <code>[first, last)</code>. Elements which are equal to value are removed from the sequence. 

**Returns**:
A <code>ForwardIterator</code> pointing to the end of the resulting range of elements which are not equal to <code>value</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/remove">https://en.cppreference.com/w/cpp/algorithm/remove</a>
* remove_if 
* remove_copy 
* remove_copy_if 

<h3 id="function-remove">
Function <code>thrust::remove</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>ForwardIterator </span><span><b>remove</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;const T & value);</span></code>
<code>remove</code> removes from the range <code>[first, last)</code> all elements that are equal to <code>value</code>. That is, <code>remove</code> returns an iterator <code>new&#95;last</code> such that the range <code>[first, new&#95;last)</code> contains no elements equal to <code>value</code>. The iterators in the range <code>[new&#95;first,last)</code> are all still dereferenceable, but the elements that they point to are unspecified. <code>remove</code> is stable, meaning that the relative order of elements that are not equal to <code>value</code> is unchanged.


The following code snippet demonstrates how to use <code>remove</code> to remove a number of interest from a range.



```cpp
#include <thrust/remove.h>
...
const int N = 6;
int A[N] = {3, 1, 4, 1, 5, 9};
int *new_end = thrust::remove(A, A + N, 1);
// The first four values of A are now {3, 4, 5, 9}
// Values beyond new_end are unspecified
```

**Note**:
The meaning of "removal" is somewhat subtle. <code>remove</code> does not destroy any iterators, and does not change the distance between <code>first</code> and <code>last</code>. (There's no way that it could do anything of the sort.) So, for example, if <code>V</code> is a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device_vector</a>, <code>remove(V.begin(), V.end(), 0)</code> does not change <code>V.size()</code>: <code>V</code> will contain just as many elements as it did before. <code>remove</code> returns an iterator that points to the end of the resulting range after elements have been removed from it; it follows that the elements after that iterator are of no interest, and may be discarded. If you are removing elements from a <a href="https://en.cppreference.com/w/cpp/container">Sequence</a>, you may simply erase them. That is, a reasonable way of removing elements from a <a href="https://en.cppreference.com/w/cpp/container">Sequence</a> is <code>S.erase(remove(S.begin(), S.end(), x), S.end())</code>.

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>, and objects of type <code>T</code> can be compared for equality with objects of <code>ForwardIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The beginning of the range of interest. 
* **`last`** The end of the range of interest. 
* **`value`** The value to remove from the range <code>[first, last)</code>. Elements which are equal to value are removed from the sequence. 

**Returns**:
A <code>ForwardIterator</code> pointing to the end of the resulting range of elements which are not equal to <code>value</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/remove">https://en.cppreference.com/w/cpp/algorithm/remove</a>
* remove_if 
* remove_copy 
* remove_copy_if 

<h3 id="function-remove-copy">
Function <code>thrust::remove&#95;copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>remove_copy</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;const T & value);</span></code>
<code>remove&#95;copy</code> copies elements that are not equal to <code>value</code> from the range <code>[first, last)</code> to a range beginning at <code>result</code>. The return value is the end of the resulting range. This operation is stable, meaning that the relative order of the elements that are copied is the same as in the range <code>[first, last)</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>remove&#95;copy</code> to copy a sequence of numbers to an output range while omitting a value of interest using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
...
const int N = 6;
int V[N] = {-2, 0, -1, 0, 1, 2};
int result[N-2];
thrust::remove_copy(thrust::host, V, V + N, result, 0);
// V remains {-2, 0, -1, 0, 1, 2}
// result is now {-2, -1, 1, 2}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>, and objects of type <code>T</code> can be compared for equality with objects of <code>InputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the range of interest. 
* **`last`** The end of the range of interest. 
* **`result`** The resulting range is copied to the sequence beginning at this location. 
* **`value`** The value to omit from the copied range. 

**Preconditions**:
The range <code>[first, last)</code> shall not overlap the range <code>[result, result + (last - first))</code>.

**Returns**:
An OutputIterator pointing to the end of the resulting range of elements which are not equal to <code>value</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/remove_copy">https://en.cppreference.com/w/cpp/algorithm/remove_copy</a>
* remove 
* remove_if 
* remove_copy_if 

<h3 id="function-remove-copy">
Function <code>thrust::remove&#95;copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>OutputIterator </span><span><b>remove_copy</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;const T & value);</span></code>
<code>remove&#95;copy</code> copies elements that are not equal to <code>value</code> from the range <code>[first, last)</code> to a range beginning at <code>result</code>. The return value is the end of the resulting range. This operation is stable, meaning that the relative order of the elements that are copied is the same as in the range <code>[first, last)</code>.


The following code snippet demonstrates how to use <code>remove&#95;copy</code> to copy a sequence of numbers to an output range while omitting a value of interest.



```cpp
#include <thrust/remove.h>
...
const int N = 6;
int V[N] = {-2, 0, -1, 0, 1, 2};
int result[N-2];
thrust::remove_copy(V, V + N, result, 0);
// V remains {-2, 0, -1, 0, 1, 2}
// result is now {-2, -1, 1, 2}
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>, and objects of type <code>T</code> can be compared for equality with objects of <code>InputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The beginning of the range of interest. 
* **`last`** The end of the range of interest. 
* **`result`** The resulting range is copied to the sequence beginning at this location. 
* **`value`** The value to omit from the copied range. 

**Preconditions**:
The range <code>[first, last)</code> shall not overlap the range <code>[result, result + (last - first))</code>.

**Returns**:
An OutputIterator pointing to the end of the resulting range of elements which are not equal to <code>value</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/remove_copy">https://en.cppreference.com/w/cpp/algorithm/remove_copy</a>
* remove 
* remove_if 
* remove_copy_if 

<h3 id="function-remove-if">
Function <code>thrust::remove&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>remove_if</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>remove&#95;if</code> removes from the range <code>[first, last)</code> every element <code>x</code> such that <code>pred(x)</code> is <code>true</code>. That is, <code>remove&#95;if</code> returns an iterator <code>new&#95;last</code> such that the range <code>[first,new&#95;last)</code> contains no elements for which <code>pred</code> is <code>true</code>. The iterators in the range <code>[new&#95;last,last)</code> are all still dereferenceable, but the elements that they point to are unspecified. <code>remove&#95;if</code> is stable, meaning that the relative order of elements that are not removed is unchanged.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>remove&#95;if</code> to remove all even numbers from an array of integers using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
...
struct is_even
{
  __host__ __device__
  bool operator()(const int x)
  {
    return (x % 2) == 0;
  }
};
...
const int N = 6;
int A[N] = {1, 4, 2, 8, 5, 7};
int *new_end = thrust::remove_if(thrust::host, A, A + N, is_even());
// The first three values of A are now {1, 5, 7}
// Values beyond new_end are unspecified
```

**Note**:
The meaning of "removal" is somewhat subtle. <code>remove&#95;if</code> does not destroy any iterators, and does not change the distance between <code>first</code> and <code>last</code>. (There's no way that it could do anything of the sort.) So, for example, if <code>V</code> is a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device_vector</a>, <code>remove&#95;if(V.begin(), V.end(), pred)</code> does not change <code>V.size()</code>: <code>V</code> will contain just as many elements as it did before. <code>remove&#95;if</code> returns an iterator that points to the end of the resulting range after elements have been removed from it; it follows that the elements after that iterator are of no interest, and may be discarded. If you are removing elements from a <a href="https://en.cppreference.com/w/cpp/container">Sequence</a>, you may simply erase them. That is, a reasonable way of removing elements from a <a href="https://en.cppreference.com/w/cpp/container">Sequence</a> is <code>S.erase(remove&#95;if(S.begin(), S.end(), pred), S.end())</code>.

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, <code>ForwardIterator</code> is mutable, and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the range of interest. 
* **`last`** The end of the range of interest. 
* **`pred`** A predicate to evaluate for each element of the range <code>[first,last)</code>. Elements for which <code>pred</code> evaluates to <code>true</code> are removed from the sequence. 

**Returns**:
A ForwardIterator pointing to the end of the resulting range of elements for which <code>pred</code> evaluated to <code>true</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/remove">https://en.cppreference.com/w/cpp/algorithm/remove</a>
* remove 
* remove_copy 
* remove_copy_if 

<h3 id="function-remove-if">
Function <code>thrust::remove&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>ForwardIterator </span><span><b>remove_if</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>remove&#95;if</code> removes from the range <code>[first, last)</code> every element <code>x</code> such that <code>pred(x)</code> is <code>true</code>. That is, <code>remove&#95;if</code> returns an iterator <code>new&#95;last</code> such that the range <code>[first,new&#95;last)</code> contains no elements for which <code>pred</code> is <code>true</code>. The iterators in the range <code>[new&#95;last,last)</code> are all still dereferenceable, but the elements that they point to are unspecified. <code>remove&#95;if</code> is stable, meaning that the relative order of elements that are not removed is unchanged.


The following code snippet demonstrates how to use <code>remove&#95;if</code> to remove all even numbers from an array of integers.



```cpp
#include <thrust/remove.h>
...
struct is_even
{
  __host__ __device__
  bool operator()(const int x)
  {
    return (x % 2) == 0;
  }
};
...
const int N = 6;
int A[N] = {1, 4, 2, 8, 5, 7};
int *new_end = thrust::remove_if(A, A + N, is_even());
// The first three values of A are now {1, 5, 7}
// Values beyond new_end are unspecified
```

**Note**:
The meaning of "removal" is somewhat subtle. <code>remove&#95;if</code> does not destroy any iterators, and does not change the distance between <code>first</code> and <code>last</code>. (There's no way that it could do anything of the sort.) So, for example, if <code>V</code> is a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device_vector</a>, <code>remove&#95;if(V.begin(), V.end(), pred)</code> does not change <code>V.size()</code>: <code>V</code> will contain just as many elements as it did before. <code>remove&#95;if</code> returns an iterator that points to the end of the resulting range after elements have been removed from it; it follows that the elements after that iterator are of no interest, and may be discarded. If you are removing elements from a <a href="https://en.cppreference.com/w/cpp/container">Sequence</a>, you may simply erase them. That is, a reasonable way of removing elements from a <a href="https://en.cppreference.com/w/cpp/container">Sequence</a> is <code>S.erase(remove&#95;if(S.begin(), S.end(), pred), S.end())</code>.

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, <code>ForwardIterator</code> is mutable, and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first`** The beginning of the range of interest. 
* **`last`** The end of the range of interest. 
* **`pred`** A predicate to evaluate for each element of the range <code>[first,last)</code>. Elements for which <code>pred</code> evaluates to <code>true</code> are removed from the sequence. 

**Returns**:
A ForwardIterator pointing to the end of the resulting range of elements for which <code>pred</code> evaluated to <code>true</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/remove">https://en.cppreference.com/w/cpp/algorithm/remove</a>
* remove 
* remove_copy 
* remove_copy_if 

<h3 id="function-remove-copy-if">
Function <code>thrust::remove&#95;copy&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>remove_copy_if</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>remove&#95;copy&#95;if</code> copies elements from the range <code>[first,last)</code> to a range beginning at <code>result</code>, except that elements for which <code>pred</code> is <code>true</code> are not copied. The return value is the end of the resulting range. This operation is stable, meaning that the relative order of the elements that are copied is the same as the range <code>[first,last)</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>remove&#95;copy&#95;if</code> to copy a sequence of numbers to an output range while omitting even numbers using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
...
struct is_even
{
  __host__ __device__
  bool operator()(const int x)
  {
    return (x % 2) == 0;
  }
};
...
const int N = 6;
int V[N] = {-2, 0, -1, 0, 1, 2};
int result[2];
thrust::remove_copy_if(thrust::host, V, V + N, result, is_even());
// V remains {-2, 0, -1, 0, 1, 2}
// result is now {-1, 1}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the range of interest. 
* **`last`** The end of the range of interest. 
* **`result`** The resulting range is copied to the sequence beginning at this location. 
* **`pred`** A predicate to evaluate for each element of the range <code>[first,last)</code>. Elements for which <code>pred</code> evaluates to <code>false</code> are not copied to the resulting sequence. 

**Preconditions**:
The range <code>[first, last)</code> shall not overlap the range <code>[result, result + (last - first))</code>.

**Returns**:
An OutputIterator pointing to the end of the resulting range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/remove_copy">https://en.cppreference.com/w/cpp/algorithm/remove_copy</a>
* remove 
* remove_copy 
* remove_if 

<h3 id="function-remove-copy-if">
Function <code>thrust::remove&#95;copy&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>OutputIterator </span><span><b>remove_copy_if</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>remove&#95;copy&#95;if</code> copies elements from the range <code>[first,last)</code> to a range beginning at <code>result</code>, except that elements for which <code>pred</code> is <code>true</code> are not copied. The return value is the end of the resulting range. This operation is stable, meaning that the relative order of the elements that are copied is the same as the range <code>[first,last)</code>.


The following code snippet demonstrates how to use <code>remove&#95;copy&#95;if</code> to copy a sequence of numbers to an output range while omitting even numbers.



```cpp
#include <thrust/remove.h>
...
struct is_even
{
  __host__ __device__
  bool operator()(const int x)
  {
    return (x % 2) == 0;
  }
};
...
const int N = 6;
int V[N] = {-2, 0, -1, 0, 1, 2};
int result[2];
thrust::remove_copy_if(V, V + N, result, is_even());
// V remains {-2, 0, -1, 0, 1, 2}
// result is now {-1, 1}
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first`** The beginning of the range of interest. 
* **`last`** The end of the range of interest. 
* **`result`** The resulting range is copied to the sequence beginning at this location. 
* **`pred`** A predicate to evaluate for each element of the range <code>[first,last)</code>. Elements for which <code>pred</code> evaluates to <code>false</code> are not copied to the resulting sequence. 

**Preconditions**:
The range <code>[first, last)</code> shall not overlap the range <code>[result, result + (last - first))</code>.

**Returns**:
An OutputIterator pointing to the end of the resulting range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/remove_copy">https://en.cppreference.com/w/cpp/algorithm/remove_copy</a>
* remove 
* remove_copy 
* remove_if 

<h3 id="function-remove-if">
Function <code>thrust::remove&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>remove_if</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator stencil,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>remove&#95;if</code> removes from the range <code>[first, last)</code> every element <code>x</code> such that <code>pred(x)</code> is <code>true</code>. That is, <code>remove&#95;if</code> returns an iterator <code>new&#95;last</code> such that the range <code>[first, new&#95;last)</code> contains no elements for which <code>pred</code> of the corresponding stencil value is <code>true</code>. The iterators in the range <code>[new&#95;last,last)</code> are all still dereferenceable, but the elements that they point to are unspecified. <code>remove&#95;if</code> is stable, meaning that the relative order of elements that are not removed is unchanged.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>remove&#95;if</code> to remove specific elements from an array of integers using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
...
const int N = 6;
int A[N] = {1, 4, 2, 8, 5, 7};
int S[N] = {0, 1, 1, 1, 0, 0};

int *new_end = thrust::remove_if(thrust::host, A, A + N, S, thrust::identity<int>());
// The first three values of A are now {1, 5, 7}
// Values beyond new_end are unspecified
```

**Note**:
The range <code>[first, last)</code> is not permitted to overlap with the range <code>[stencil, stencil + (last - first))</code>.

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a> and <code>ForwardIterator</code> is mutable. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the range of interest. 
* **`last`** The end of the range of interest. 
* **`stencil`** The beginning of the stencil sequence. 
* **`pred`** A predicate to evaluate for each element of the range <code>[stencil, stencil + (last - first))</code>. Elements for which <code>pred</code> evaluates to <code>true</code> are removed from the sequence <code>[first, last)</code>

**Preconditions**:
* The range <code>[first, last)</code> shall not overlap the range <code>[result, result + (last - first))</code>. 
* The range <code>[stencil, stencil + (last - first))</code> shall not overlap the range <code>[result, result + (last - first))</code>.

**Returns**:
A ForwardIterator pointing to the end of the resulting range of elements for which <code>pred</code> evaluated to <code>true</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/remove">https://en.cppreference.com/w/cpp/algorithm/remove</a>
* remove 
* remove_copy 
* remove_copy_if 

<h3 id="function-remove-if">
Function <code>thrust::remove&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>ForwardIterator </span><span><b>remove_if</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator stencil,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>remove&#95;if</code> removes from the range <code>[first, last)</code> every element <code>x</code> such that <code>pred(x)</code> is <code>true</code>. That is, <code>remove&#95;if</code> returns an iterator <code>new&#95;last</code> such that the range <code>[first, new&#95;last)</code> contains no elements for which <code>pred</code> of the corresponding stencil value is <code>true</code>. The iterators in the range <code>[new&#95;last,last)</code> are all still dereferenceable, but the elements that they point to are unspecified. <code>remove&#95;if</code> is stable, meaning that the relative order of elements that are not removed is unchanged.


The following code snippet demonstrates how to use <code>remove&#95;if</code> to remove specific elements from an array of integers.



```cpp
#include <thrust/remove.h>
...
const int N = 6;
int A[N] = {1, 4, 2, 8, 5, 7};
int S[N] = {0, 1, 1, 1, 0, 0};

int *new_end = thrust::remove_if(A, A + N, S, thrust::identity<int>());
// The first three values of A are now {1, 5, 7}
// Values beyond new_end are unspecified
```

**Note**:
The range <code>[first, last)</code> is not permitted to overlap with the range <code>[stencil, stencil + (last - first))</code>.

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a> and <code>ForwardIterator</code> is mutable. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first`** The beginning of the range of interest. 
* **`last`** The end of the range of interest. 
* **`stencil`** The beginning of the stencil sequence. 
* **`pred`** A predicate to evaluate for each element of the range <code>[stencil, stencil + (last - first))</code>. Elements for which <code>pred</code> evaluates to <code>true</code> are removed from the sequence <code>[first, last)</code>

**Preconditions**:
* The range <code>[first, last)</code> shall not overlap the range <code>[result, result + (last - first))</code>. 
* The range <code>[stencil, stencil + (last - first))</code> shall not overlap the range <code>[result, result + (last - first))</code>.

**Returns**:
A ForwardIterator pointing to the end of the resulting range of elements for which <code>pred</code> evaluated to <code>true</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/remove">https://en.cppreference.com/w/cpp/algorithm/remove</a>
* remove 
* remove_copy 
* remove_copy_if 

<h3 id="function-remove-copy-if">
Function <code>thrust::remove&#95;copy&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>remove_copy_if</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>remove&#95;copy&#95;if</code> copies elements from the range <code>[first,last)</code> to a range beginning at <code>result</code>, except that elements for which <code>pred</code> of the corresponding stencil value is <code>true</code> are not copied. The return value is the end of the resulting range. This operation is stable, meaning that the relative order of the elements that are copied is the same as the range <code>[first,last)</code>.

The algorithm's execution policy is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>remove&#95;copy&#95;if</code> to copy a sequence of numbers to an output range while omitting specific elements using the <code>thrust::host</code> execution policy for parallelization.



```cpp
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
...
const int N = 6;
int V[N] = {-2, 0, -1, 0, 1, 2};
int S[N] = { 1, 1,  0, 1, 0, 1};
int result[2];
thrust::remove_copy_if(thrust::host, V, V + N, S, result, thrust::identity<int>());
// V remains {-2, 0, -1, 0, 1, 2}
// result is now {-1, 1}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the range of interest. 
* **`last`** The end of the range of interest. 
* **`stencil`** The beginning of the stencil sequence. 
* **`result`** The resulting range is copied to the sequence beginning at this location. 
* **`pred`** A predicate to evaluate for each element of the range <code>[first,last)</code>. Elements for which <code>pred</code> evaluates to <code>false</code> are not copied to the resulting sequence. 

**Preconditions**:
The range <code>[stencil, stencil + (last - first))</code> shall not overlap the range <code>[result, result + (last - first))</code>.

**Returns**:
An OutputIterator pointing to the end of the resulting range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/remove_copy">https://en.cppreference.com/w/cpp/algorithm/remove_copy</a>
* remove 
* remove_copy 
* remove_if 
* copy_if 

<h3 id="function-remove-copy-if">
Function <code>thrust::remove&#95;copy&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>OutputIterator </span><span><b>remove_copy_if</b>(InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>remove&#95;copy&#95;if</code> copies elements from the range <code>[first,last)</code> to a range beginning at <code>result</code>, except that elements for which <code>pred</code> of the corresponding stencil value is <code>true</code> are not copied. The return value is the end of the resulting range. This operation is stable, meaning that the relative order of the elements that are copied is the same as the range <code>[first,last)</code>.


The following code snippet demonstrates how to use <code>remove&#95;copy&#95;if</code> to copy a sequence of numbers to an output range while omitting specific elements.



```cpp
#include <thrust/remove.h>
...
const int N = 6;
int V[N] = {-2, 0, -1, 0, 1, 2};
int S[N] = { 1, 1,  0, 1, 0, 1};
int result[2];
thrust::remove_copy_if(V, V + N, S, result, thrust::identity<int>());
// V remains {-2, 0, -1, 0, 1, 2}
// result is now {-1, 1}
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, <code>InputIterator1's</code><code>value&#95;type</code> is convertible to a type in <code>OutputIterator's</code> set of <code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first`** The beginning of the range of interest. 
* **`last`** The end of the range of interest. 
* **`stencil`** The beginning of the stencil sequence. 
* **`result`** The resulting range is copied to the sequence beginning at this location. 
* **`pred`** A predicate to evaluate for each element of the range <code>[first,last)</code>. Elements for which <code>pred</code> evaluates to <code>false</code> are not copied to the resulting sequence. 

**Preconditions**:
The range <code>[stencil, stencil + (last - first))</code> shall not overlap the range <code>[result, result + (last - first))</code>.

**Returns**:
An OutputIterator pointing to the end of the resulting range.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/remove_copy">https://en.cppreference.com/w/cpp/algorithm/remove_copy</a>
* remove 
* remove_copy 
* remove_if 
* copy_if 

<h3 id="function-unique">
Function <code>thrust::unique</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>unique</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span></code>
For each group of consecutive elements in the range <code>[first, last)</code> with the same value, <code>unique</code> removes all but the first element of the group. The return value is an iterator <code>new&#95;last</code> such that no two consecutive elements in the range <code>[first, new&#95;last)</code> are equal. The iterators in the range <code>[new&#95;last, last)</code> are all still dereferenceable, but the elements that they point to are unspecified. <code>unique</code> is stable, meaning that the relative order of elements that are not removed is unchanged.

This version of <code>unique</code> uses <code>operator==</code> to test for equality.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>unique</code> to compact a sequence of numbers to remove consecutive duplicates using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1};
int *new_end = thrust::unique(thrust::host, A, A + N);
// The first four values of A are now {1, 3, 2, 1}
// Values beyond new_end are unspecified.
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable, and <code>ForwardIterator's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the input range. 
* **`last`** The end of the input range. 

**Returns**:
The end of the unique range <code>[first, new&#95;last)</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/unique">https://en.cppreference.com/w/cpp/algorithm/unique</a>
* unique_copy 

<h3 id="function-unique">
Function <code>thrust::unique</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator&gt;</span>
<span>ForwardIterator </span><span><b>unique</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span></code>
For each group of consecutive elements in the range <code>[first, last)</code> with the same value, <code>unique</code> removes all but the first element of the group. The return value is an iterator <code>new&#95;last</code> such that no two consecutive elements in the range <code>[first, new&#95;last)</code> are equal. The iterators in the range <code>[new&#95;last, last)</code> are all still dereferenceable, but the elements that they point to are unspecified. <code>unique</code> is stable, meaning that the relative order of elements that are not removed is unchanged.

This version of <code>unique</code> uses <code>operator==</code> to test for equality.


The following code snippet demonstrates how to use <code>unique</code> to compact a sequence of numbers to remove consecutive duplicates.



```cpp
#include <thrust/unique.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1};
int *new_end = thrust::unique(A, A + N);
// The first four values of A are now {1, 3, 2, 1}
// Values beyond new_end are unspecified.
```

**Template Parameters**:
**`ForwardIterator`**: is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable, and <code>ForwardIterator's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>.

**Function Parameters**:
* **`first`** The beginning of the input range. 
* **`last`** The end of the input range. 

**Returns**:
The end of the unique range <code>[first, new&#95;last)</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/unique">https://en.cppreference.com/w/cpp/algorithm/unique</a>
* unique_copy 

<h3 id="function-unique">
Function <code>thrust::unique</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>unique</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span></code>
For each group of consecutive elements in the range <code>[first, last)</code> with the same value, <code>unique</code> removes all but the first element of the group. The return value is an iterator <code>new&#95;last</code> such that no two consecutive elements in the range <code>[first, new&#95;last)</code> are equal. The iterators in the range <code>[new&#95;last, last)</code> are all still dereferenceable, but the elements that they point to are unspecified. <code>unique</code> is stable, meaning that the relative order of elements that are not removed is unchanged.

This version of <code>unique</code> uses the function object <code>binary&#95;pred</code> to test for equality.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>unique</code> to compact a sequence of numbers to remove consecutive duplicates using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1};
int *new_end = thrust::unique(thrust::host, A, A + N, thrust::equal_to<int>());
// The first four values of A are now {1, 3, 2, 1}
// Values beyond new_end are unspecified.
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable, and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to <code>BinaryPredicate's</code><code>first&#95;argument&#95;type</code> and to <code>BinaryPredicate's</code><code>second&#95;argument&#95;type</code>. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the input range. 
* **`last`** The end of the input range. 
* **`binary_pred`** The binary predicate used to determine equality. 

**Returns**:
The end of the unique range <code>[first, new&#95;last)</code>

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/unique">https://en.cppreference.com/w/cpp/algorithm/unique</a>
* unique_copy 

<h3 id="function-unique">
Function <code>thrust::unique</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>ForwardIterator </span><span><b>unique</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span></code>
For each group of consecutive elements in the range <code>[first, last)</code> with the same value, <code>unique</code> removes all but the first element of the group. The return value is an iterator <code>new&#95;last</code> such that no two consecutive elements in the range <code>[first, new&#95;last)</code> are equal. The iterators in the range <code>[new&#95;last, last)</code> are all still dereferenceable, but the elements that they point to are unspecified. <code>unique</code> is stable, meaning that the relative order of elements that are not removed is unchanged.

This version of <code>unique</code> uses the function object <code>binary&#95;pred</code> to test for equality.


The following code snippet demonstrates how to use <code>unique</code> to compact a sequence of numbers to remove consecutive duplicates.



```cpp
#include <thrust/unique.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1};
int *new_end = thrust::unique(A, A + N, thrust::equal_to<int>());
// The first four values of A are now {1, 3, 2, 1}
// Values beyond new_end are unspecified.
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable, and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to <code>BinaryPredicate's</code><code>first&#95;argument&#95;type</code> and to <code>BinaryPredicate's</code><code>second&#95;argument&#95;type</code>. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.

**Function Parameters**:
* **`first`** The beginning of the input range. 
* **`last`** The end of the input range. 
* **`binary_pred`** The binary predicate used to determine equality. 

**Returns**:
The end of the unique range <code>[first, new&#95;last)</code>

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/unique">https://en.cppreference.com/w/cpp/algorithm/unique</a>
* unique_copy 

<h3 id="function-unique-copy">
Function <code>thrust::unique&#95;copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>unique_copy</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>unique&#95;copy</code> copies elements from the range <code>[first, last)</code> to a range beginning with <code>result</code>, except that in a consecutive group of duplicate elements only the first one is copied. The return value is the end of the range to which the elements are copied.

The reason there are two different versions of unique_copy is that there are two different definitions of what it means for a consecutive group of elements to be duplicates. In the first version, the test is simple equality: the elements in a range <code>[f, l)</code> are duplicates if, for every iterator <code>i</code> in the range, either <code>i == f</code> or else <code>&#42;i == &#42;(i-1)</code>. In the second, the test is an arbitrary <code>BinaryPredicate</code><code>binary&#95;pred:</code> the elements in <code>[f, l)</code> are duplicates if, for every iterator <code>i</code> in the range, either <code>i == f</code> or else <code>binary&#95;pred(&#42;i, &#42;(i-1))</code> is <code>true</code>.

This version of <code>unique&#95;copy</code> uses <code>operator==</code> to test for equality.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>unique&#95;copy</code> to compact a sequence of numbers to remove consecutive duplicates using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1};
int B[N];
int *result_end = thrust::unique_copy(thrust::host, A, A + N, B);
// The first four values of B are now {1, 3, 2, 1} and (result_end - B) is 4
// Values beyond result_end are unspecified
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the input range. 
* **`last`** The end of the input range. 
* **`result`** The beginning of the output range. 

**Preconditions**:
The range <code>[first,last)</code> and the range <code>[result, result + (last - first))</code> shall not overlap.

**Returns**:
The end of the unique range <code>[result, result&#95;end)</code>.

**See**:
* unique 
* <a href="https://en.cppreference.com/w/cpp/algorithm/unique_copy">https://en.cppreference.com/w/cpp/algorithm/unique_copy</a>

<h3 id="function-unique-copy">
Function <code>thrust::unique&#95;copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b>unique_copy</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>unique&#95;copy</code> copies elements from the range <code>[first, last)</code> to a range beginning with <code>result</code>, except that in a consecutive group of duplicate elements only the first one is copied. The return value is the end of the range to which the elements are copied.

The reason there are two different versions of unique_copy is that there are two different definitions of what it means for a consecutive group of elements to be duplicates. In the first version, the test is simple equality: the elements in a range <code>[f, l)</code> are duplicates if, for every iterator <code>i</code> in the range, either <code>i == f</code> or else <code>&#42;i == &#42;(i-1)</code>. In the second, the test is an arbitrary <code>BinaryPredicate</code><code>binary&#95;pred:</code> the elements in <code>[f, l)</code> are duplicates if, for every iterator <code>i</code> in the range, either <code>i == f</code> or else <code>binary&#95;pred(&#42;i, &#42;(i-1))</code> is <code>true</code>.

This version of <code>unique&#95;copy</code> uses <code>operator==</code> to test for equality.


The following code snippet demonstrates how to use <code>unique&#95;copy</code> to compact a sequence of numbers to remove consecutive duplicates.



```cpp
#include <thrust/unique.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1};
int B[N];
int *result_end = thrust::unique_copy(A, A + N, B);
// The first four values of B are now {1, 3, 2, 1} and (result_end - B) is 4
// Values beyond result_end are unspecified
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The beginning of the input range. 
* **`last`** The end of the input range. 
* **`result`** The beginning of the output range. 

**Preconditions**:
The range <code>[first,last)</code> and the range <code>[result, result + (last - first))</code> shall not overlap.

**Returns**:
The end of the unique range <code>[result, result&#95;end)</code>.

**See**:
* unique 
* <a href="https://en.cppreference.com/w/cpp/algorithm/unique_copy">https://en.cppreference.com/w/cpp/algorithm/unique_copy</a>

<h3 id="function-unique-copy">
Function <code>thrust::unique&#95;copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>unique_copy</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span></code>
<code>unique&#95;copy</code> copies elements from the range <code>[first, last)</code> to a range beginning with <code>result</code>, except that in a consecutive group of duplicate elements only the first one is copied. The return value is the end of the range to which the elements are copied.

This version of <code>unique&#95;copy</code> uses the function object <code>binary&#95;pred</code> to test for equality.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>unique&#95;copy</code> to compact a sequence of numbers to remove consecutive duplicates using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1};
int B[N];
int *result_end = thrust::unique_copy(thrust::host, A, A + N, B, thrust::equal_to<int>());
// The first four values of B are now {1, 3, 2, 1} and (result_end - B) is 4
// Values beyond result_end are unspecified.
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the input range. 
* **`last`** The end of the input range. 
* **`result`** The beginning of the output range. 
* **`binary_pred`** The binary predicate used to determine equality. 

**Preconditions**:
The range <code>[first,last)</code> and the range <code>[result, result + (last - first))</code> shall not overlap.

**Returns**:
The end of the unique range <code>[result, result&#95;end)</code>.

**See**:
* unique 
* <a href="https://en.cppreference.com/w/cpp/algorithm/unique_copy">https://en.cppreference.com/w/cpp/algorithm/unique_copy</a>

<h3 id="function-unique-copy">
Function <code>thrust::unique&#95;copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>OutputIterator </span><span><b>unique_copy</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span></code>
<code>unique&#95;copy</code> copies elements from the range <code>[first, last)</code> to a range beginning with <code>result</code>, except that in a consecutive group of duplicate elements only the first one is copied. The return value is the end of the range to which the elements are copied.

This version of <code>unique&#95;copy</code> uses the function object <code>binary&#95;pred</code> to test for equality.


The following code snippet demonstrates how to use <code>unique&#95;copy</code> to compact a sequence of numbers to remove consecutive duplicates.



```cpp
#include <thrust/unique.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1};
int B[N];
int *result_end = thrust::unique_copy(A, A + N, B, thrust::equal_to<int>());
// The first four values of B are now {1, 3, 2, 1} and (result_end - B) is 4
// Values beyond result_end are unspecified.
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.

**Function Parameters**:
* **`first`** The beginning of the input range. 
* **`last`** The end of the input range. 
* **`result`** The beginning of the output range. 
* **`binary_pred`** The binary predicate used to determine equality. 

**Preconditions**:
The range <code>[first,last)</code> and the range <code>[result, result + (last - first))</code> shall not overlap.

**Returns**:
The end of the unique range <code>[result, result&#95;end)</code>.

**See**:
* unique 
* <a href="https://en.cppreference.com/w/cpp/algorithm/unique_copy">https://en.cppreference.com/w/cpp/algorithm/unique_copy</a>

<h3 id="function-unique-by-key">
Function <code>thrust::unique&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator1,</span>
<span>&nbsp;&nbsp;typename ForwardIterator2&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< ForwardIterator1, ForwardIterator2 > </span><span><b>unique_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator1 keys_first,</span>
<span>&nbsp;&nbsp;ForwardIterator1 keys_last,</span>
<span>&nbsp;&nbsp;ForwardIterator2 values_first);</span></code>
<code>unique&#95;by&#95;key</code> is a generalization of <code>unique</code> to key-value pairs. For each group of consecutive keys in the range <code>[keys&#95;first, keys&#95;last)</code> that are equal, <code>unique&#95;by&#95;key</code> removes all but the first element of the group. Similarly, the corresponding values in the range <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> are also removed.

The return value is a <code>pair</code> of iterators <code>(new&#95;keys&#95;last,new&#95;values&#95;last)</code> such that no two consecutive elements in the range <code>[keys&#95;first, new&#95;keys&#95;last)</code> are equal.

This version of <code>unique&#95;by&#95;key</code> uses <code>operator==</code> to test for equality and <code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1project1st.html">project1st</a></code> to reduce values with equal keys.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>unique&#95;by&#95;key</code> to compact a sequence of key/value pairs to remove consecutive duplicates using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1}; // keys
int B[N] = {9, 8, 7, 6, 5, 4, 3}; // values

thrust::pair<int*,int*> new_end;
new_end = thrust::unique_by_key(thrust::host, A, A + N, B);

// The first four keys in A are now {1, 3, 2, 1} and new_end.first - A is 4.
// The first four values in B are now {9, 8, 5, 3} and new_end.second - B is 4.
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator1</code> is mutable, and <code>ForwardIterator's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>. 
* **`ForwardIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator2</code> is mutable.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`keys_first`** The beginning of the key range. 
* **`keys_last`** The end of the key range. 
* **`values_first`** The beginning of the value range. 

**Preconditions**:
The range <code>[keys&#95;first, keys&#95;last)</code> and the range <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> shall not overlap.

**Returns**:
A pair of iterators at end of the ranges <code>[key&#95;first, keys&#95;new&#95;last)</code> and <code>[values&#95;first, values&#95;new&#95;last)</code>.

**See**:
* unique 
* unique_by_key_copy 
* reduce_by_key 

<h3 id="function-unique-by-key">
Function <code>thrust::unique&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator1,</span>
<span>&nbsp;&nbsp;typename ForwardIterator2&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< ForwardIterator1, ForwardIterator2 > </span><span><b>unique_by_key</b>(ForwardIterator1 keys_first,</span>
<span>&nbsp;&nbsp;ForwardIterator1 keys_last,</span>
<span>&nbsp;&nbsp;ForwardIterator2 values_first);</span></code>
<code>unique&#95;by&#95;key</code> is a generalization of <code>unique</code> to key-value pairs. For each group of consecutive keys in the range <code>[keys&#95;first, keys&#95;last)</code> that are equal, <code>unique&#95;by&#95;key</code> removes all but the first element of the group. Similarly, the corresponding values in the range <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> are also removed.

The return value is a <code>pair</code> of iterators <code>(new&#95;keys&#95;last,new&#95;values&#95;last)</code> such that no two consecutive elements in the range <code>[keys&#95;first, new&#95;keys&#95;last)</code> are equal.

This version of <code>unique&#95;by&#95;key</code> uses <code>operator==</code> to test for equality and <code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1project1st.html">project1st</a></code> to reduce values with equal keys.


The following code snippet demonstrates how to use <code>unique&#95;by&#95;key</code> to compact a sequence of key/value pairs to remove consecutive duplicates.



```cpp
#include <thrust/unique.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1}; // keys
int B[N] = {9, 8, 7, 6, 5, 4, 3}; // values

thrust::pair<int*,int*> new_end;
new_end = thrust::unique_by_key(A, A + N, B);

// The first four keys in A are now {1, 3, 2, 1} and new_end.first - A is 4.
// The first four values in B are now {9, 8, 5, 3} and new_end.second - B is 4.
```

**Template Parameters**:
* **`ForwardIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator1</code> is mutable, and <code>ForwardIterator's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>. 
* **`ForwardIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator2</code> is mutable.

**Function Parameters**:
* **`keys_first`** The beginning of the key range. 
* **`keys_last`** The end of the key range. 
* **`values_first`** The beginning of the value range. 

**Preconditions**:
The range <code>[keys&#95;first, keys&#95;last)</code> and the range <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> shall not overlap.

**Returns**:
A pair of iterators at end of the ranges <code>[key&#95;first, keys&#95;new&#95;last)</code> and <code>[values&#95;first, values&#95;new&#95;last)</code>.

**See**:
* unique 
* unique_by_key_copy 
* reduce_by_key 

<h3 id="function-unique-by-key">
Function <code>thrust::unique&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator1,</span>
<span>&nbsp;&nbsp;typename ForwardIterator2,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< ForwardIterator1, ForwardIterator2 > </span><span><b>unique_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator1 keys_first,</span>
<span>&nbsp;&nbsp;ForwardIterator1 keys_last,</span>
<span>&nbsp;&nbsp;ForwardIterator2 values_first,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span></code>
<code>unique&#95;by&#95;key</code> is a generalization of <code>unique</code> to key-value pairs. For each group of consecutive keys in the range <code>[keys&#95;first, keys&#95;last)</code> that are equal, <code>unique&#95;by&#95;key</code> removes all but the first element of the group. Similarly, the corresponding values in the range <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> are also removed.

This version of <code>unique&#95;by&#95;key</code> uses the function object <code>binary&#95;pred</code> to test for equality and <code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1project1st.html">project1st</a></code> to reduce values with equal keys.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>unique&#95;by&#95;key</code> to compact a sequence of key/value pairs to remove consecutive duplicates using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1}; // keys
int B[N] = {9, 8, 7, 6, 5, 4, 3}; // values

thrust::pair<int*,int*> new_end;
thrust::equal_to<int> binary_pred;
new_end = thrust::unique_by_key(thrust::host, keys, keys + N, values, binary_pred);

// The first four keys in A are now {1, 3, 2, 1} and new_end.first - A is 4.
// The first four values in B are now {9, 8, 5, 3} and new_end.second - B is 4.
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator1</code> is mutable, and <code>ForwardIterator's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>. 
* **`ForwardIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator2</code> is mutable. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`keys_first`** The beginning of the key range. 
* **`keys_last`** The end of the key range. 
* **`values_first`** The beginning of the value range. 
* **`binary_pred`** The binary predicate used to determine equality. 

**Preconditions**:
The range <code>[keys&#95;first, keys&#95;last)</code> and the range <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> shall not overlap.

**Returns**:
The end of the unique range <code>[first, new&#95;last)</code>.

**See**:
* unique 
* unique_by_key_copy 
* reduce_by_key 

<h3 id="function-unique-by-key">
Function <code>thrust::unique&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator1,</span>
<span>&nbsp;&nbsp;typename ForwardIterator2,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< ForwardIterator1, ForwardIterator2 > </span><span><b>unique_by_key</b>(ForwardIterator1 keys_first,</span>
<span>&nbsp;&nbsp;ForwardIterator1 keys_last,</span>
<span>&nbsp;&nbsp;ForwardIterator2 values_first,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span></code>
<code>unique&#95;by&#95;key</code> is a generalization of <code>unique</code> to key-value pairs. For each group of consecutive keys in the range <code>[keys&#95;first, keys&#95;last)</code> that are equal, <code>unique&#95;by&#95;key</code> removes all but the first element of the group. Similarly, the corresponding values in the range <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> are also removed.

This version of <code>unique&#95;by&#95;key</code> uses the function object <code>binary&#95;pred</code> to test for equality and <code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1project1st.html">project1st</a></code> to reduce values with equal keys.


The following code snippet demonstrates how to use <code>unique&#95;by&#95;key</code> to compact a sequence of key/value pairs to remove consecutive duplicates.



```cpp
#include <thrust/unique.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1}; // keys
int B[N] = {9, 8, 7, 6, 5, 4, 3}; // values

thrust::pair<int*,int*> new_end;
thrust::equal_to<int> binary_pred;
new_end = thrust::unique_by_key(keys, keys + N, values, binary_pred);

// The first four keys in A are now {1, 3, 2, 1} and new_end.first - A is 4.
// The first four values in B are now {9, 8, 5, 3} and new_end.second - B is 4.
```

**Template Parameters**:
* **`ForwardIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator1</code> is mutable, and <code>ForwardIterator's</code><code>value&#95;type</code> is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>. 
* **`ForwardIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator2</code> is mutable. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.

**Function Parameters**:
* **`keys_first`** The beginning of the key range. 
* **`keys_last`** The end of the key range. 
* **`values_first`** The beginning of the value range. 
* **`binary_pred`** The binary predicate used to determine equality. 

**Preconditions**:
The range <code>[keys&#95;first, keys&#95;last)</code> and the range <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> shall not overlap.

**Returns**:
The end of the unique range <code>[first, new&#95;last)</code>.

**See**:
* unique 
* unique_by_key_copy 
* reduce_by_key 

<h3 id="function-unique-by-key-copy">
Function <code>thrust::unique&#95;by&#95;key&#95;copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>unique_by_key_copy</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last,</span>
<span>&nbsp;&nbsp;InputIterator2 values_first,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result);</span></code>
<code>unique&#95;by&#95;key&#95;copy</code> is a generalization of <code>unique&#95;copy</code> to key-value pairs. For each group of consecutive keys in the range <code>[keys&#95;first, keys&#95;last)</code> that are equal, <code>unique&#95;by&#95;key&#95;copy</code> copies the first element of the group to a range beginning with <code>keys&#95;result</code> and the corresponding values from the range <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> are copied to a range beginning with <code>values&#95;result</code>.

This version of <code>unique&#95;by&#95;key&#95;copy</code> uses <code>operator==</code> to test for equality and <code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1project1st.html">project1st</a></code> to reduce values with equal keys.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>unique&#95;by&#95;key&#95;copy</code> to compact a sequence of key/value pairs and with equal keys using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
int C[N];                         // output keys
int D[N];                         // output values

thrust::pair<int*,int*> new_end;
new_end = thrust::unique_by_key_copy(thrust::host, A, A + N, B, C, D);

// The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
// The first four values in D are now {9, 8, 5, 3} and new_end.second - D is 4.
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and and <code>InputIterator1's</code><code>value&#95;type</code> is convertible to <code>OutputIterator1's</code><code>value&#95;type</code>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>OutputIterator2's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`keys_first`** The beginning of the input key range. 
* **`keys_last`** The end of the input key range. 
* **`values_first`** The beginning of the input value range. 
* **`keys_result`** The beginning of the output key range. 
* **`values_result`** The beginning of the output value range. 

**Preconditions**:
The input ranges shall not overlap either output range.

**Returns**:
A pair of iterators at end of the ranges <code>[keys&#95;result, keys&#95;result&#95;last)</code> and <code>[values&#95;result, values&#95;result&#95;last)</code>.

**See**:
* unique_copy 
* unique_by_key 
* reduce_by_key 

<h3 id="function-unique-by-key-copy">
Function <code>thrust::unique&#95;by&#95;key&#95;copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>unique_by_key_copy</b>(InputIterator1 keys_first,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last,</span>
<span>&nbsp;&nbsp;InputIterator2 values_first,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result);</span></code>
<code>unique&#95;by&#95;key&#95;copy</code> is a generalization of <code>unique&#95;copy</code> to key-value pairs. For each group of consecutive keys in the range <code>[keys&#95;first, keys&#95;last)</code> that are equal, <code>unique&#95;by&#95;key&#95;copy</code> copies the first element of the group to a range beginning with <code>keys&#95;result</code> and the corresponding values from the range <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> are copied to a range beginning with <code>values&#95;result</code>.

This version of <code>unique&#95;by&#95;key&#95;copy</code> uses <code>operator==</code> to test for equality and <code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1project1st.html">project1st</a></code> to reduce values with equal keys.


The following code snippet demonstrates how to use <code>unique&#95;by&#95;key&#95;copy</code> to compact a sequence of key/value pairs and with equal keys.



```cpp
#include <thrust/unique.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
int C[N];                         // output keys
int D[N];                         // output values

thrust::pair<int*,int*> new_end;
new_end = thrust::unique_by_key_copy(A, A + N, B, C, D);

// The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
// The first four values in D are now {9, 8, 5, 3} and new_end.second - D is 4.
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and and <code>InputIterator1's</code><code>value&#95;type</code> is convertible to <code>OutputIterator1's</code><code>value&#95;type</code>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>OutputIterator2's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`keys_first`** The beginning of the input key range. 
* **`keys_last`** The end of the input key range. 
* **`values_first`** The beginning of the input value range. 
* **`keys_result`** The beginning of the output key range. 
* **`values_result`** The beginning of the output value range. 

**Preconditions**:
The input ranges shall not overlap either output range.

**Returns**:
A pair of iterators at end of the ranges <code>[keys&#95;result, keys&#95;result&#95;last)</code> and <code>[values&#95;result, values&#95;result&#95;last)</code>.

**See**:
* unique_copy 
* unique_by_key 
* reduce_by_key 

<h3 id="function-unique-by-key-copy">
Function <code>thrust::unique&#95;by&#95;key&#95;copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>unique_by_key_copy</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last,</span>
<span>&nbsp;&nbsp;InputIterator2 values_first,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span></code>
<code>unique&#95;by&#95;key&#95;copy</code> is a generalization of <code>unique&#95;copy</code> to key-value pairs. For each group of consecutive keys in the range <code>[keys&#95;first, keys&#95;last)</code> that are equal, <code>unique&#95;by&#95;key&#95;copy</code> copies the first element of the group to a range beginning with <code>keys&#95;result</code> and the corresponding values from the range <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> are copied to a range beginning with <code>values&#95;result</code>.

This version of <code>unique&#95;by&#95;key&#95;copy</code> uses the function object <code>binary&#95;pred</code> to test for equality and <code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1project1st.html">project1st</a></code> to reduce values with equal keys.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>unique&#95;by&#95;key&#95;copy</code> to compact a sequence of key/value pairs and with equal keys using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
int C[N];                         // output keys
int D[N];                         // output values

thrust::pair<int*,int*> new_end;
thrust::equal_to<int> binary_pred;
new_end = thrust::unique_by_key_copy(thrust::host, A, A + N, B, C, D, binary_pred);

// The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
// The first four values in D are now {9, 8, 5, 3} and new_end.second - D is 4.
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and and <code>InputIterator1's</code><code>value&#95;type</code> is convertible to <code>OutputIterator1's</code><code>value&#95;type</code>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>OutputIterator2's</code><code>value&#95;type</code>. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`keys_first`** The beginning of the input key range. 
* **`keys_last`** The end of the input key range. 
* **`values_first`** The beginning of the input value range. 
* **`keys_result`** The beginning of the output key range. 
* **`values_result`** The beginning of the output value range. 
* **`binary_pred`** The binary predicate used to determine equality. 

**Preconditions**:
The input ranges shall not overlap either output range.

**Returns**:
A pair of iterators at end of the ranges <code>[keys&#95;result, keys&#95;result&#95;last)</code> and <code>[values&#95;result, values&#95;result&#95;last)</code>.

**See**:
* unique_copy 
* unique_by_key 
* reduce_by_key 

<h3 id="function-unique-by-key-copy">
Function <code>thrust::unique&#95;by&#95;key&#95;copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>unique_by_key_copy</b>(InputIterator1 keys_first,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last,</span>
<span>&nbsp;&nbsp;InputIterator2 values_first,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_result,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_result,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span></code>
<code>unique&#95;by&#95;key&#95;copy</code> is a generalization of <code>unique&#95;copy</code> to key-value pairs. For each group of consecutive keys in the range <code>[keys&#95;first, keys&#95;last)</code> that are equal, <code>unique&#95;by&#95;key&#95;copy</code> copies the first element of the group to a range beginning with <code>keys&#95;result</code> and the corresponding values from the range <code>[values&#95;first, values&#95;first + (keys&#95;last - keys&#95;first))</code> are copied to a range beginning with <code>values&#95;result</code>.

This version of <code>unique&#95;by&#95;key&#95;copy</code> uses the function object <code>binary&#95;pred</code> to test for equality and <code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1project1st.html">project1st</a></code> to reduce values with equal keys.


The following code snippet demonstrates how to use <code>unique&#95;by&#95;key&#95;copy</code> to compact a sequence of key/value pairs and with equal keys.



```cpp
#include <thrust/unique.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
int C[N];                         // output keys
int D[N];                         // output values

thrust::pair<int*,int*> new_end;
thrust::equal_to<int> binary_pred;
new_end = thrust::unique_by_key_copy(A, A + N, B, C, D, binary_pred);

// The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
// The first four values in D are now {9, 8, 5, 3} and new_end.second - D is 4.
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and and <code>InputIterator1's</code><code>value&#95;type</code> is convertible to <code>OutputIterator1's</code><code>value&#95;type</code>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>OutputIterator2's</code><code>value&#95;type</code>. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.

**Function Parameters**:
* **`keys_first`** The beginning of the input key range. 
* **`keys_last`** The end of the input key range. 
* **`values_first`** The beginning of the input value range. 
* **`keys_result`** The beginning of the output key range. 
* **`values_result`** The beginning of the output value range. 
* **`binary_pred`** The binary predicate used to determine equality. 

**Preconditions**:
The input ranges shall not overlap either output range.

**Returns**:
A pair of iterators at end of the ranges <code>[keys&#95;result, keys&#95;result&#95;last)</code> and <code>[values&#95;result, values&#95;result&#95;last)</code>.

**See**:
* unique_copy 
* unique_by_key 
* reduce_by_key 

<h3 id="function-unique-count">
Function <code>thrust::unique&#95;count</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">thrust::iterator_traits</a>< ForwardIterator >::difference_type </span><span><b>unique_count</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span></code>
<code>unique&#95;count</code> counts runs of equal elements in the range <code>[first, last)</code> with the same value,

This version of <code>unique&#95;count</code> uses the function object <code>binary&#95;pred</code> to test for equality.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>unique&#95;count</code> to determine a number of runs of equal elements using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1};
int count = thrust::unique_count(thrust::host, A, A + N, thrust::equal_to<int>());
// count is now 4
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to <code>BinaryPredicate's</code><code>first&#95;argument&#95;type</code> and to <code>BinaryPredicate's</code><code>second&#95;argument&#95;type</code>. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the input range. 
* **`last`** The end of the input range. 
* **`binary_pred`** The binary predicate used to determine equality. 

**Returns**:
The number of runs of equal elements in <code>[first, new&#95;last)</code>

**See**:
* unique_copy 
* unique_by_key_copy 
* reduce_by_key_copy 

<h3 id="function-unique-count">
Function <code>thrust::unique&#95;count</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">thrust::iterator_traits</a>< ForwardIterator >::difference_type </span><span><b>unique_count</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span></code>
<code>unique&#95;count</code> counts runs of equal elements in the range <code>[first, last)</code> with the same value,

This version of <code>unique&#95;count</code> uses <code>operator==</code> to test for equality.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>unique&#95;count</code> to determine the number of runs of equal elements using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1};
int count = thrust::unique_count(thrust::host, A, A + N);
// count is now 4
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to <code>BinaryPredicate's</code><code>first&#95;argument&#95;type</code> and to <code>BinaryPredicate's</code><code>second&#95;argument&#95;type</code>. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the input range. 
* **`last`** The end of the input range. 
* **`binary_pred`** The binary predicate used to determine equality. 

**Returns**:
The number of runs of equal elements in <code>[first, new&#95;last)</code>

**See**:
* unique_copy 
* unique_by_key_copy 
* reduce_by_key_copy 

<h3 id="function-unique-count">
Function <code>thrust::unique&#95;count</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">thrust::iterator_traits</a>< ForwardIterator >::difference_type </span><span><b>unique_count</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span></code>
<code>unique&#95;count</code> counts runs of equal elements in the range <code>[first, last)</code> with the same value,

This version of <code>unique&#95;count</code> uses the function object <code>binary&#95;pred</code> to test for equality.


The following code snippet demonstrates how to use <code>unique&#95;count</code> to determine the number of runs of equal elements:



```cpp
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1};
int count = thrust::unique_count(A, A + N, thrust::equal_to<int>());
// count is now 4
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to <code>BinaryPredicate's</code><code>first&#95;argument&#95;type</code> and to <code>BinaryPredicate's</code><code>second&#95;argument&#95;type</code>. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the input range. 
* **`last`** The end of the input range. 
* **`binary_pred`** The binary predicate used to determine equality. 

**Returns**:
The number of runs of equal elements in <code>[first, new&#95;last)</code>

**See**:
* unique_copy 
* unique_by_key_copy 
* reduce_by_key_copy 

<h3 id="function-unique-count">
Function <code>thrust::unique&#95;count</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">thrust::iterator_traits</a>< ForwardIterator >::difference_type </span><span><b>unique_count</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last);</span></code>
<code>unique&#95;count</code> counts runs of equal elements in the range <code>[first, last)</code> with the same value,

This version of <code>unique&#95;count</code> uses <code>operator==</code> to test for equality.


The following code snippet demonstrates how to use <code>unique&#95;count</code> to determine the number of runs of equal elements:



```cpp
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1};
int count = thrust::unique_count(thrust::host, A, A + N);
// count is now 4
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to <code>BinaryPredicate's</code><code>first&#95;argument&#95;type</code> and to <code>BinaryPredicate's</code><code>second&#95;argument&#95;type</code>. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the input range. 
* **`last`** The end of the input range. 
* **`binary_pred`** The binary predicate used to determine equality. 

**Returns**:
The number of runs of equal elements in <code>[first, new&#95;last)</code>

**See**:
* unique_copy 
* unique_by_key_copy 
* reduce_by_key_copy 


