---
title: Segmented Prefix Sums
parent: Prefix Sums
grand_parent: Algorithms
nav_exclude: false
has_children: true
has_toc: false
---

# Segmented Prefix Sums

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__segmentedprefixsums.html#function-inclusive-scan-by-key">thrust::inclusive&#95;scan&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__segmentedprefixsums.html#function-inclusive-scan-by-key">thrust::inclusive&#95;scan&#95;by&#95;key</a></b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__segmentedprefixsums.html#function-inclusive-scan-by-key">thrust::inclusive&#95;scan&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__segmentedprefixsums.html#function-inclusive-scan-by-key">thrust::inclusive&#95;scan&#95;by&#95;key</a></b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate,</span>
<span>&nbsp;&nbsp;typename AssociativeOperator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__segmentedprefixsums.html#function-inclusive-scan-by-key">thrust::inclusive&#95;scan&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred,</span>
<span>&nbsp;&nbsp;AssociativeOperator binary_op);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate,</span>
<span>&nbsp;&nbsp;typename AssociativeOperator&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__segmentedprefixsums.html#function-inclusive-scan-by-key">thrust::inclusive&#95;scan&#95;by&#95;key</a></b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred,</span>
<span>&nbsp;&nbsp;AssociativeOperator binary_op);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__segmentedprefixsums.html#function-exclusive-scan-by-key">thrust::exclusive&#95;scan&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__segmentedprefixsums.html#function-exclusive-scan-by-key">thrust::exclusive&#95;scan&#95;by&#95;key</a></b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__segmentedprefixsums.html#function-exclusive-scan-by-key">thrust::exclusive&#95;scan&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;T init);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__segmentedprefixsums.html#function-exclusive-scan-by-key">thrust::exclusive&#95;scan&#95;by&#95;key</a></b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;T init);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__segmentedprefixsums.html#function-exclusive-scan-by-key">thrust::exclusive&#95;scan&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;T init,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__segmentedprefixsums.html#function-exclusive-scan-by-key">thrust::exclusive&#95;scan&#95;by&#95;key</a></b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;T init,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate,</span>
<span>&nbsp;&nbsp;typename AssociativeOperator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__segmentedprefixsums.html#function-exclusive-scan-by-key">thrust::exclusive&#95;scan&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;T init,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred,</span>
<span>&nbsp;&nbsp;AssociativeOperator binary_op);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate,</span>
<span>&nbsp;&nbsp;typename AssociativeOperator&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__segmentedprefixsums.html#function-exclusive-scan-by-key">thrust::exclusive&#95;scan&#95;by&#95;key</a></b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;T init,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred,</span>
<span>&nbsp;&nbsp;AssociativeOperator binary_op);</span>
</code>

## Functions

<h3 id="function-inclusive-scan-by-key">
Function <code>thrust::inclusive&#95;scan&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>inclusive_scan_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>inclusive&#95;scan&#95;by&#95;key</code> computes an inclusive key-value or 'segmented' prefix sum operation. The term 'inclusive' means that each result includes the corresponding input operand in the partial sum. The term 'segmented' means that the partial sums are broken into distinct segments. In other words, within each segment a separate inclusive scan operation is computed. Refer to the code sample below for example usage.

This version of <code>inclusive&#95;scan&#95;by&#95;key</code> assumes <code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1equal__to.html">equal&#95;to</a></code> as the binary predicate used to compare adjacent keys. Specifically, consecutive iterators <code>i</code> and <code>i+1</code> in the range <code>[first1, last1)</code> belong to the same segment if <code>&#42;i == &#42;(i+1)</code>, and belong to different segments otherwise.

This version of <code>inclusive&#95;scan&#95;by&#95;key</code> assumes <code>plus</code> as the associative operator used to perform the prefix sum. When the input and output sequences are the same, the scan is performed in-place.

Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>inclusive&#95;scan&#95;by&#95;key</code> using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
...

int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};

thrust::inclusive_scan_by_key(thrust::host, keys, keys + 10, data, data); // in-place scan

// data is now {1, 2, 3, 1, 2, 1, 1, 2, 3, 4};
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>, and if <code>x</code> and <code>y</code> are objects of <code>OutputIterator's</code><code>value&#95;type</code>, then <code>binary&#95;op(x,y)</code> is defined.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the key sequence. 
* **`last1`** The end of the key sequence. 
* **`first2`** The beginning of the input value sequence. 
* **`result`** The beginning of the output value sequence. 

**Preconditions**:
* <code>first1</code> may equal <code>result</code> but the range <code>[first1, last1)</code> and the range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise. 
* <code>first2</code> may equal <code>result</code> but the range <code>[first2, first2 + (last1 - first1)</code> and range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
* inclusive_scan 
* exclusive_scan_by_key 

<h3 id="function-inclusive-scan-by-key">
Function <code>thrust::inclusive&#95;scan&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b>inclusive_scan_by_key</b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>inclusive&#95;scan&#95;by&#95;key</code> computes an inclusive key-value or 'segmented' prefix sum operation. The term 'inclusive' means that each result includes the corresponding input operand in the partial sum. The term 'segmented' means that the partial sums are broken into distinct segments. In other words, within each segment a separate inclusive scan operation is computed. Refer to the code sample below for example usage.

This version of <code>inclusive&#95;scan&#95;by&#95;key</code> assumes <code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1equal__to.html">equal&#95;to</a></code> as the binary predicate used to compare adjacent keys. Specifically, consecutive iterators <code>i</code> and <code>i+1</code> in the range <code>[first1, last1)</code> belong to the same segment if <code>&#42;i == &#42;(i+1)</code>, and belong to different segments otherwise.

This version of <code>inclusive&#95;scan&#95;by&#95;key</code> assumes <code>plus</code> as the associative operator used to perform the prefix sum. When the input and output sequences are the same, the scan is performed in-place.

Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run.


The following code snippet demonstrates how to use <code>inclusive&#95;scan&#95;by&#95;key</code>



```cpp
#include <thrust/scan.h>

int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};

thrust::inclusive_scan_by_key(keys, keys + 10, data, data); // in-place scan

// data is now {1, 2, 3, 1, 2, 1, 1, 2, 3, 4};
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>, and if <code>x</code> and <code>y</code> are objects of <code>OutputIterator's</code><code>value&#95;type</code>, then <code>binary&#95;op(x,y)</code> is defined.

**Function Parameters**:
* **`first1`** The beginning of the key sequence. 
* **`last1`** The end of the key sequence. 
* **`first2`** The beginning of the input value sequence. 
* **`result`** The beginning of the output value sequence. 

**Preconditions**:
* <code>first1</code> may equal <code>result</code> but the range <code>[first1, last1)</code> and the range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise. 
* <code>first2</code> may equal <code>result</code> but the range <code>[first2, first2 + (last1 - first1)</code> and range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
* inclusive_scan 
* exclusive_scan_by_key 

<h3 id="function-inclusive-scan-by-key">
Function <code>thrust::inclusive&#95;scan&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>inclusive_scan_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span></code>
<code>inclusive&#95;scan&#95;by&#95;key</code> computes an inclusive key-value or 'segmented' prefix sum operation. The term 'inclusive' means that each result includes the corresponding input operand in the partial sum. The term 'segmented' means that the partial sums are broken into distinct segments. In other words, within each segment a separate inclusive scan operation is computed. Refer to the code sample below for example usage.

This version of <code>inclusive&#95;scan&#95;by&#95;key</code> uses the binary predicate <code>pred</code> to compare adjacent keys. Specifically, consecutive iterators <code>i</code> and <code>i+1</code> in the range <code>[first1, last1)</code> belong to the same segment if <code>binary&#95;pred(&#42;i, &#42;(i+1))</code> is true, and belong to different segments otherwise.

This version of <code>inclusive&#95;scan&#95;by&#95;key</code> assumes <code>plus</code> as the associative operator used to perform the prefix sum. When the input and output sequences are the same, the scan is performed in-place.

Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>inclusive&#95;scan&#95;by&#95;key</code> using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...

int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};

thrust::equal_to<int> binary_pred;

thrust::inclusive_scan_by_key(thrust::host, keys, keys + 10, data, data, binary_pred); // in-place scan

// data is now {1, 2, 3, 1, 2, 1, 1, 2, 3, 4};
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>, and if <code>x</code> and <code>y</code> are objects of <code>OutputIterator's</code><code>value&#95;type</code>, then <code>binary&#95;op(x,y)</code> is defined. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the key sequence. 
* **`last1`** The end of the key sequence. 
* **`first2`** The beginning of the input value sequence. 
* **`result`** The beginning of the output value sequence. 
* **`binary_pred`** The binary predicate used to determine equality of keys. 

**Preconditions**:
* <code>first1</code> may equal <code>result</code> but the range <code>[first1, last1)</code> and the range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise. 
* <code>first2</code> may equal <code>result</code> but the range <code>[first2, first2 + (last1 - first1)</code> and range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
* inclusive_scan 
* exclusive_scan_by_key 

<h3 id="function-inclusive-scan-by-key">
Function <code>thrust::inclusive&#95;scan&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>OutputIterator </span><span><b>inclusive_scan_by_key</b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span></code>
<code>inclusive&#95;scan&#95;by&#95;key</code> computes an inclusive key-value or 'segmented' prefix sum operation. The term 'inclusive' means that each result includes the corresponding input operand in the partial sum. The term 'segmented' means that the partial sums are broken into distinct segments. In other words, within each segment a separate inclusive scan operation is computed. Refer to the code sample below for example usage.

This version of <code>inclusive&#95;scan&#95;by&#95;key</code> uses the binary predicate <code>pred</code> to compare adjacent keys. Specifically, consecutive iterators <code>i</code> and <code>i+1</code> in the range <code>[first1, last1)</code> belong to the same segment if <code>binary&#95;pred(&#42;i, &#42;(i+1))</code> is true, and belong to different segments otherwise.

This version of <code>inclusive&#95;scan&#95;by&#95;key</code> assumes <code>plus</code> as the associative operator used to perform the prefix sum. When the input and output sequences are the same, the scan is performed in-place.

Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run.


The following code snippet demonstrates how to use <code>inclusive&#95;scan&#95;by&#95;key</code>



```cpp
#include <thrust/scan.h>
#include <thrust/functional.h>

int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};

thrust::equal_to<int> binary_pred;

thrust::inclusive_scan_by_key(keys, keys + 10, data, data, binary_pred); // in-place scan

// data is now {1, 2, 3, 1, 2, 1, 1, 2, 3, 4};
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>, and if <code>x</code> and <code>y</code> are objects of <code>OutputIterator's</code><code>value&#95;type</code>, then <code>binary&#95;op(x,y)</code> is defined. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>.

**Function Parameters**:
* **`first1`** The beginning of the key sequence. 
* **`last1`** The end of the key sequence. 
* **`first2`** The beginning of the input value sequence. 
* **`result`** The beginning of the output value sequence. 
* **`binary_pred`** The binary predicate used to determine equality of keys. 

**Preconditions**:
* <code>first1</code> may equal <code>result</code> but the range <code>[first1, last1)</code> and the range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise. 
* <code>first2</code> may equal <code>result</code> but the range <code>[first2, first2 + (last1 - first1)</code> and range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
* inclusive_scan 
* exclusive_scan_by_key 

<h3 id="function-inclusive-scan-by-key">
Function <code>thrust::inclusive&#95;scan&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate,</span>
<span>&nbsp;&nbsp;typename AssociativeOperator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>inclusive_scan_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred,</span>
<span>&nbsp;&nbsp;AssociativeOperator binary_op);</span></code>
<code>inclusive&#95;scan&#95;by&#95;key</code> computes an inclusive key-value or 'segmented' prefix sum operation. The term 'inclusive' means that each result includes the corresponding input operand in the partial sum. The term 'segmented' means that the partial sums are broken into distinct segments. In other words, within each segment a separate inclusive scan operation is computed. Refer to the code sample below for example usage.

This version of <code>inclusive&#95;scan&#95;by&#95;key</code> uses the binary predicate <code>pred</code> to compare adjacent keys. Specifically, consecutive iterators <code>i</code> and <code>i+1</code> in the range <code>[first1, last1)</code> belong to the same segment if <code>binary&#95;pred(&#42;i, &#42;(i+1))</code> is true, and belong to different segments otherwise.

This version of <code>inclusive&#95;scan&#95;by&#95;key</code> uses the associative operator <code>binary&#95;op</code> to perform the prefix sum. When the input and output sequences are the same, the scan is performed in-place.

Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>inclusive&#95;scan&#95;by&#95;key</code> using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...

int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};

thrust::equal_to<int> binary_pred;
thrust::plus<int>     binary_op;

thrust::inclusive_scan_by_key(thrust::host, keys, keys + 10, data, data, binary_pred, binary_op); // in-place scan

// data is now {1, 2, 3, 1, 2, 1, 1, 2, 3, 4};
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>, and if <code>x</code> and <code>y</code> are objects of <code>OutputIterator's</code><code>value&#95;type</code>, then <code>binary&#95;op(x,y)</code> is defined. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>. 
* **`AssociativeOperator`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a> and <code>AssociativeOperator's</code><code>result&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the key sequence. 
* **`last1`** The end of the key sequence. 
* **`first2`** The beginning of the input value sequence. 
* **`result`** The beginning of the output value sequence. 
* **`binary_pred`** The binary predicate used to determine equality of keys. 
* **`binary_op`** The associatve operator used to 'sum' values. 

**Preconditions**:
* <code>first1</code> may equal <code>result</code> but the range <code>[first1, last1)</code> and the range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise. 
* <code>first2</code> may equal <code>result</code> but the range <code>[first2, first2 + (last1 - first1)</code> and range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
* inclusive_scan 
* exclusive_scan_by_key 

<h3 id="function-inclusive-scan-by-key">
Function <code>thrust::inclusive&#95;scan&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate,</span>
<span>&nbsp;&nbsp;typename AssociativeOperator&gt;</span>
<span>OutputIterator </span><span><b>inclusive_scan_by_key</b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred,</span>
<span>&nbsp;&nbsp;AssociativeOperator binary_op);</span></code>
<code>inclusive&#95;scan&#95;by&#95;key</code> computes an inclusive key-value or 'segmented' prefix sum operation. The term 'inclusive' means that each result includes the corresponding input operand in the partial sum. The term 'segmented' means that the partial sums are broken into distinct segments. In other words, within each segment a separate inclusive scan operation is computed. Refer to the code sample below for example usage.

This version of <code>inclusive&#95;scan&#95;by&#95;key</code> uses the binary predicate <code>pred</code> to compare adjacent keys. Specifically, consecutive iterators <code>i</code> and <code>i+1</code> in the range <code>[first1, last1)</code> belong to the same segment if <code>binary&#95;pred(&#42;i, &#42;(i+1))</code> is true, and belong to different segments otherwise.

Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run.

This version of <code>inclusive&#95;scan&#95;by&#95;key</code> uses the associative operator <code>binary&#95;op</code> to perform the prefix sum. When the input and output sequences are the same, the scan is performed in-place.


The following code snippet demonstrates how to use <code>inclusive&#95;scan&#95;by&#95;key</code>



```cpp
#include <thrust/scan.h>
#include <thrust/functional.h>

int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};

thrust::equal_to<int> binary_pred;
thrust::plus<int>     binary_op;

thrust::inclusive_scan_by_key(keys, keys + 10, data, data, binary_pred, binary_op); // in-place scan

// data is now {1, 2, 3, 1, 2, 1, 1, 2, 3, 4};
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>, and if <code>x</code> and <code>y</code> are objects of <code>OutputIterator's</code><code>value&#95;type</code>, then <code>binary&#95;op(x,y)</code> is defined. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>. 
* **`AssociativeOperator`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a> and <code>AssociativeOperator's</code><code>result&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first1`** The beginning of the key sequence. 
* **`last1`** The end of the key sequence. 
* **`first2`** The beginning of the input value sequence. 
* **`result`** The beginning of the output value sequence. 
* **`binary_pred`** The binary predicate used to determine equality of keys. 
* **`binary_op`** The associatve operator used to 'sum' values. 

**Preconditions**:
* <code>first1</code> may equal <code>result</code> but the range <code>[first1, last1)</code> and the range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise. 
* <code>first2</code> may equal <code>result</code> but the range <code>[first2, first2 + (last1 - first1)</code> and range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
* inclusive_scan 
* exclusive_scan_by_key 

<h3 id="function-exclusive-scan-by-key">
Function <code>thrust::exclusive&#95;scan&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>exclusive_scan_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>exclusive&#95;scan&#95;by&#95;key</code> computes an exclusive segmented prefix

This version of <code>exclusive&#95;scan&#95;by&#95;key</code> uses the value <code>0</code> to initialize the exclusive scan operation.

This version of <code>exclusive&#95;scan&#95;by&#95;key</code> assumes <code>plus</code> as the associative operator used to perform the prefix sum. When the input and output sequences are the same, the scan is performed in-place.

This version of <code>exclusive&#95;scan&#95;by&#95;key</code> assumes <code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1equal__to.html">equal&#95;to</a></code> as the binary predicate used to compare adjacent keys. Specifically, consecutive iterators <code>i</code> and <code>i+1</code> in the range <code>[first1, last1</code> belong to the same segment if <code>&#42;i == &#42;(i+1)</code>, and belong to different segments otherwise.

Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run.

Refer to the most general form of <code>exclusive&#95;scan&#95;by&#95;key</code> for additional details.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>exclusive&#95;scan&#95;by&#95;key</code> using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
...

int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
int vals[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

thrust::exclusive_scan_by_key(thrust::host, key, key + 10, vals, vals); // in-place scan

// vals is now {0, 1, 2, 0, 1, 0, 0, 1, 2, 3};
```

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the key sequence. 
* **`last1`** The end of the key sequence. 
* **`first2`** The beginning of the input value sequence. 
* **`result`** The beginning of the output value sequence.

**Preconditions**:
* <code>first1</code> may equal <code>result</code> but the range <code>[first1, last1)</code> and the range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise. 
* <code>first2</code> may equal <code>result</code> but the range <code>[first2, first2 + (last1 - first1)</code> and range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise.

**See**:
exclusive_scan 

<h3 id="function-exclusive-scan-by-key">
Function <code>thrust::exclusive&#95;scan&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b>exclusive_scan_by_key</b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>exclusive&#95;scan&#95;by&#95;key</code> computes an exclusive segmented prefix

This version of <code>exclusive&#95;scan&#95;by&#95;key</code> uses the value <code>0</code> to initialize the exclusive scan operation.

This version of <code>exclusive&#95;scan&#95;by&#95;key</code> assumes <code>plus</code> as the associative operator used to perform the prefix sum. When the input and output sequences are the same, the scan is performed in-place.

This version of <code>exclusive&#95;scan&#95;by&#95;key</code> assumes <code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1equal__to.html">equal&#95;to</a></code> as the binary predicate used to compare adjacent keys. Specifically, consecutive iterators <code>i</code> and <code>i+1</code> in the range <code>[first1, last1</code> belong to the same segment if <code>&#42;i == &#42;(i+1)</code>, and belong to different segments otherwise.

Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run.

Refer to the most general form of <code>exclusive&#95;scan&#95;by&#95;key</code> for additional details.


The following code snippet demonstrates how to use <code>exclusive&#95;scan&#95;by&#95;key</code>.



```cpp
#include <thrust/scan.h>

int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
int vals[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

thrust::exclusive_scan_by_key(key, key + 10, vals, vals); // in-place scan

// vals is now {0, 1, 2, 0, 1, 0, 0, 1, 2, 3};
```

**Function Parameters**:
* **`first1`** The beginning of the key sequence. 
* **`last1`** The end of the key sequence. 
* **`first2`** The beginning of the input value sequence. 
* **`result`** The beginning of the output value sequence.

**Preconditions**:
* <code>first1</code> may equal <code>result</code> but the range <code>[first1, last1)</code> and the range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise. 
* <code>first2</code> may equal <code>result</code> but the range <code>[first2, first2 + (last1 - first1)</code> and range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise.

**See**:
exclusive_scan 

<h3 id="function-exclusive-scan-by-key">
Function <code>thrust::exclusive&#95;scan&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>exclusive_scan_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;T init);</span></code>
<code>exclusive&#95;scan&#95;by&#95;key</code> computes an exclusive key-value or 'segmented' prefix sum operation. The term 'exclusive' means that each result does not include the corresponding input operand in the partial sum. The term 'segmented' means that the partial sums are broken into distinct segments. In other words, within each segment a separate exclusive scan operation is computed. Refer to the code sample below for example usage.

This version of <code>exclusive&#95;scan&#95;by&#95;key</code> uses the value <code>init</code> to initialize the exclusive scan operation.

Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>exclusive&#95;scan&#95;by&#95;key</code> using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...

int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
int vals[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

int init = 5;

thrust::exclusive_scan_by_key(thrust::host, key, key + 10, vals, vals, init); // in-place scan

// vals is now {5, 6, 7, 5, 6, 5, 5, 6, 7, 8};
```

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the key sequence. 
* **`last1`** The end of the key sequence. 
* **`first2`** The beginning of the input value sequence. 
* **`result`** The beginning of the output value sequence. 
* **`init`** The initial of the exclusive sum value. 

**Preconditions**:
* <code>first1</code> may equal <code>result</code> but the range <code>[first1, last1)</code> and the range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise. 
* <code>first2</code> may equal <code>result</code> but the range <code>[first2, first2 + (last1 - first1)</code> and range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
* exclusive_scan 
* inclusive_scan_by_key 

<h3 id="function-exclusive-scan-by-key">
Function <code>thrust::exclusive&#95;scan&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>OutputIterator </span><span><b>exclusive_scan_by_key</b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;T init);</span></code>
<code>exclusive&#95;scan&#95;by&#95;key</code> computes an exclusive key-value or 'segmented' prefix sum operation. The term 'exclusive' means that each result does not include the corresponding input operand in the partial sum. The term 'segmented' means that the partial sums are broken into distinct segments. In other words, within each segment a separate exclusive scan operation is computed. Refer to the code sample below for example usage.

This version of <code>exclusive&#95;scan&#95;by&#95;key</code> uses the value <code>init</code> to initialize the exclusive scan operation.

Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run.


The following code snippet demonstrates how to use <code>exclusive&#95;scan&#95;by&#95;key</code>



```cpp
#include <thrust/scan.h>
#include <thrust/functional.h>

int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
int vals[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

int init = 5;

thrust::exclusive_scan_by_key(key, key + 10, vals, vals, init); // in-place scan

// vals is now {5, 6, 7, 5, 6, 5, 5, 6, 7, 8};
```

**Function Parameters**:
* **`first1`** The beginning of the key sequence. 
* **`last1`** The end of the key sequence. 
* **`first2`** The beginning of the input value sequence. 
* **`result`** The beginning of the output value sequence. 
* **`init`** The initial of the exclusive sum value. 

**Preconditions**:
* <code>first1</code> may equal <code>result</code> but the range <code>[first1, last1)</code> and the range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise. 
* <code>first2</code> may equal <code>result</code> but the range <code>[first2, first2 + (last1 - first1)</code> and range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
* exclusive_scan 
* inclusive_scan_by_key 

<h3 id="function-exclusive-scan-by-key">
Function <code>thrust::exclusive&#95;scan&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>exclusive_scan_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;T init,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span></code>
<code>exclusive&#95;scan&#95;by&#95;key</code> computes an exclusive key-value or 'segmented' prefix sum operation. The term 'exclusive' means that each result does not include the corresponding input operand in the partial sum. The term 'segmented' means that the partial sums are broken into distinct segments. In other words, within each segment a separate exclusive scan operation is computed. Refer to the code sample below for example usage.

This version of <code>exclusive&#95;scan&#95;by&#95;key</code> uses the value <code>init</code> to initialize the exclusive scan operation.

This version of <code>exclusive&#95;scan&#95;by&#95;key</code> uses the binary predicate <code>binary&#95;pred</code> to compare adjacent keys. Specifically, consecutive iterators <code>i</code> and <code>i+1</code> in the range <code>[first1, last1)</code> belong to the same segment if <code>binary&#95;pred(&#42;i, &#42;(i+1))</code> is true, and belong to different segments otherwise.

Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>exclusive&#95;scan&#95;by&#95;key</code> using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...

int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
int vals[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

int init = 5;

thrust::equal_to<int> binary_pred;

thrust::exclusive_scan_by_key(thrust::host, key, key + 10, vals, vals, init, binary_pred); // in-place scan

// vals is now {5, 6, 7, 5, 6, 5, 5, 6, 7, 8};
```

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the key sequence. 
* **`last1`** The end of the key sequence. 
* **`first2`** The beginning of the input value sequence. 
* **`result`** The beginning of the output value sequence. 
* **`init`** The initial of the exclusive sum value. 
* **`binary_pred`** The binary predicate used to determine equality of keys. 

**Preconditions**:
* <code>first1</code> may equal <code>result</code> but the range <code>[first1, last1)</code> and the range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise. 
* <code>first2</code> may equal <code>result</code> but the range <code>[first2, first2 + (last1 - first1)</code> and range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
* exclusive_scan 
* inclusive_scan_by_key 

<h3 id="function-exclusive-scan-by-key">
Function <code>thrust::exclusive&#95;scan&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>OutputIterator </span><span><b>exclusive_scan_by_key</b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;T init,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span></code>
<code>exclusive&#95;scan&#95;by&#95;key</code> computes an exclusive key-value or 'segmented' prefix sum operation. The term 'exclusive' means that each result does not include the corresponding input operand in the partial sum. The term 'segmented' means that the partial sums are broken into distinct segments. In other words, within each segment a separate exclusive scan operation is computed. Refer to the code sample below for example usage.

This version of <code>exclusive&#95;scan&#95;by&#95;key</code> uses the value <code>init</code> to initialize the exclusive scan operation.

This version of <code>exclusive&#95;scan&#95;by&#95;key</code> uses the binary predicate <code>binary&#95;pred</code> to compare adjacent keys. Specifically, consecutive iterators <code>i</code> and <code>i+1</code> in the range <code>[first1, last1)</code> belong to the same segment if <code>binary&#95;pred(&#42;i, &#42;(i+1))</code> is true, and belong to different segments otherwise.

Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run.


The following code snippet demonstrates how to use <code>exclusive&#95;scan&#95;by&#95;key</code>



```cpp
#include <thrust/scan.h>
#include <thrust/functional.h>

int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
int vals[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

int init = 5;

thrust::equal_to<int> binary_pred;

thrust::exclusive_scan_by_key(key, key + 10, vals, vals, init, binary_pred); // in-place scan

// vals is now {5, 6, 7, 5, 6, 5, 5, 6, 7, 8};
```

**Function Parameters**:
* **`first1`** The beginning of the key sequence. 
* **`last1`** The end of the key sequence. 
* **`first2`** The beginning of the input value sequence. 
* **`result`** The beginning of the output value sequence. 
* **`init`** The initial of the exclusive sum value. 
* **`binary_pred`** The binary predicate used to determine equality of keys. 

**Preconditions**:
* <code>first1</code> may equal <code>result</code> but the range <code>[first1, last1)</code> and the range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise. 
* <code>first2</code> may equal <code>result</code> but the range <code>[first2, first2 + (last1 - first1)</code> and range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
* exclusive_scan 
* inclusive_scan_by_key 

<h3 id="function-exclusive-scan-by-key">
Function <code>thrust::exclusive&#95;scan&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate,</span>
<span>&nbsp;&nbsp;typename AssociativeOperator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>exclusive_scan_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;T init,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred,</span>
<span>&nbsp;&nbsp;AssociativeOperator binary_op);</span></code>
<code>exclusive&#95;scan&#95;by&#95;key</code> computes an exclusive key-value or 'segmented' prefix sum operation. The term 'exclusive' means that each result does not include the corresponding input operand in the partial sum. The term 'segmented' means that the partial sums are broken into distinct segments. In other words, within each segment a separate exclusive scan operation is computed. Refer to the code sample below for example usage.

This version of <code>exclusive&#95;scan&#95;by&#95;key</code> uses the value <code>init</code> to initialize the exclusive scan operation.

This version of <code>exclusive&#95;scan&#95;by&#95;key</code> uses the binary predicate <code>binary&#95;pred</code> to compare adjacent keys. Specifically, consecutive iterators <code>i</code> and <code>i+1</code> in the range <code>[first1, last1)</code> belong to the same segment if <code>binary&#95;pred(&#42;i, &#42;(i+1))</code> is true, and belong to different segments otherwise.

This version of <code>exclusive&#95;scan&#95;by&#95;key</code> uses the associative operator <code>binary&#95;op</code> to perform the prefix sum. When the input and output sequences are the same, the scan is performed in-place.

Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>exclusive&#95;scan&#95;by&#95;key</code> using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...

int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
int vals[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

int init = 5;

thrust::equal_to<int> binary_pred;
thrust::plus<int>     binary_op;

thrust::exclusive_scan_by_key(thrust::host, key, key + 10, vals, vals, init, binary_pred, binary_op); // in-place scan

// vals is now {5, 6, 7, 5, 6, 5, 5, 6, 7, 8};
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>, and if <code>x</code> and <code>y</code> are objects of <code>OutputIterator's</code><code>value&#95;type</code>, then <code>binary&#95;op(x,y)</code> is defined. 
* **`T`** is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>. 
* **`AssociativeOperator`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a> and <code>AssociativeOperator's</code><code>result&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first1`** The beginning of the key sequence. 
* **`last1`** The end of the key sequence. 
* **`first2`** The beginning of the input value sequence. 
* **`result`** The beginning of the output value sequence. 
* **`init`** The initial of the exclusive sum value. 
* **`binary_pred`** The binary predicate used to determine equality of keys. 
* **`binary_op`** The associatve operator used to 'sum' values. 

**Preconditions**:
* <code>first1</code> may equal <code>result</code> but the range <code>[first1, last1)</code> and the range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise. 
* <code>first2</code> may equal <code>result</code> but the range <code>[first2, first2 + (last1 - first1)</code> and range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
* exclusive_scan 
* inclusive_scan_by_key 

<h3 id="function-exclusive-scan-by-key">
Function <code>thrust::exclusive&#95;scan&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate,</span>
<span>&nbsp;&nbsp;typename AssociativeOperator&gt;</span>
<span>OutputIterator </span><span><b>exclusive_scan_by_key</b>(InputIterator1 first1,</span>
<span>&nbsp;&nbsp;InputIterator1 last1,</span>
<span>&nbsp;&nbsp;InputIterator2 first2,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;T init,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred,</span>
<span>&nbsp;&nbsp;AssociativeOperator binary_op);</span></code>
<code>exclusive&#95;scan&#95;by&#95;key</code> computes an exclusive key-value or 'segmented' prefix sum operation. The term 'exclusive' means that each result does not include the corresponding input operand in the partial sum. The term 'segmented' means that the partial sums are broken into distinct segments. In other words, within each segment a separate exclusive scan operation is computed. Refer to the code sample below for example usage.

This version of <code>exclusive&#95;scan&#95;by&#95;key</code> uses the value <code>init</code> to initialize the exclusive scan operation.

This version of <code>exclusive&#95;scan&#95;by&#95;key</code> uses the binary predicate <code>binary&#95;pred</code> to compare adjacent keys. Specifically, consecutive iterators <code>i</code> and <code>i+1</code> in the range <code>[first1, last1)</code> belong to the same segment if <code>binary&#95;pred(&#42;i, &#42;(i+1))</code> is true, and belong to different segments otherwise.

This version of <code>exclusive&#95;scan&#95;by&#95;key</code> uses the associative operator <code>binary&#95;op</code> to perform the prefix sum. When the input and output sequences are the same, the scan is performed in-place.

Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run.


The following code snippet demonstrates how to use <code>exclusive&#95;scan&#95;by&#95;key</code>



```cpp
#include <thrust/scan.h>
#include <thrust/functional.h>

int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
int vals[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

int init = 5;

thrust::equal_to<int> binary_pred;
thrust::plus<int>     binary_op;

thrust::exclusive_scan_by_key(key, key + 10, vals, vals, init, binary_pred, binary_op); // in-place scan

// vals is now {5, 6, 7, 5, 6, 5, 5, 6, 7, 8};
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>, and if <code>x</code> and <code>y</code> are objects of <code>OutputIterator's</code><code>value&#95;type</code>, then <code>binary&#95;op(x,y)</code> is defined. 
* **`T`** is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>. 
* **`AssociativeOperator`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a> and <code>AssociativeOperator's</code><code>result&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first1`** The beginning of the key sequence. 
* **`last1`** The end of the key sequence. 
* **`first2`** The beginning of the input value sequence. 
* **`result`** The beginning of the output value sequence. 
* **`init`** The initial of the exclusive sum value. 
* **`binary_pred`** The binary predicate used to determine equality of keys. 
* **`binary_op`** The associatve operator used to 'sum' values. 

**Preconditions**:
* <code>first1</code> may equal <code>result</code> but the range <code>[first1, last1)</code> and the range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise. 
* <code>first2</code> may equal <code>result</code> but the range <code>[first2, first2 + (last1 - first1)</code> and range <code>[result, result + (last1 - first1))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
* exclusive_scan 
* inclusive_scan_by_key 


