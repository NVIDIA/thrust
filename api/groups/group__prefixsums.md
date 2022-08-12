---
title: Prefix Sums
parent: Algorithms
grand_parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Prefix Sums

## Groups

* **[Segmented Prefix Sums]({{ site.baseurl }}/api/groups/group__segmentedprefixsums.html)**
* **[Transformed Prefix Sums]({{ site.baseurl }}/api/groups/group__transformed__prefixsums.html)**

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__prefixsums.html#function-inclusive-scan">thrust::inclusive&#95;scan</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__prefixsums.html#function-inclusive-scan">thrust::inclusive&#95;scan</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename AssociativeOperator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__prefixsums.html#function-inclusive-scan">thrust::inclusive&#95;scan</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;AssociativeOperator binary_op);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename AssociativeOperator&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__prefixsums.html#function-inclusive-scan">thrust::inclusive&#95;scan</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;AssociativeOperator binary_op);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__prefixsums.html#function-exclusive-scan">thrust::exclusive&#95;scan</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__prefixsums.html#function-exclusive-scan">thrust::exclusive&#95;scan</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__prefixsums.html#function-exclusive-scan">thrust::exclusive&#95;scan</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;T init);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__prefixsums.html#function-exclusive-scan">thrust::exclusive&#95;scan</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;T init);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename AssociativeOperator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__prefixsums.html#function-exclusive-scan">thrust::exclusive&#95;scan</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;T init,</span>
<span>&nbsp;&nbsp;AssociativeOperator binary_op);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename AssociativeOperator&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__prefixsums.html#function-exclusive-scan">thrust::exclusive&#95;scan</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;T init,</span>
<span>&nbsp;&nbsp;AssociativeOperator binary_op);</span>
</code>

## Functions

<h3 id="function-inclusive-scan">
Function <code>thrust::inclusive&#95;scan</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>inclusive_scan</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>inclusive&#95;scan</code> computes an inclusive prefix sum operation. The term 'inclusive' means that each result includes the corresponding input operand in the partial sum. More precisely, <code>&#42;first</code> is assigned to <code>&#42;result</code> and the sum of <code>&#42;first</code> and <code>&#42;(first + 1)</code> is assigned to <code>&#42;(result + 1)</code>, and so on. This version of <code>inclusive&#95;scan</code> assumes plus as the associative operator. 

 When the input and output sequences are the same, the scan is performed in-place.

<code>inclusive&#95;scan</code> is similar to <code>std::partial&#95;sum</code> in the STL. The primary difference between the two functions is that <code>std::partial&#95;sum</code> guarantees a serial summation order, while <code>inclusive&#95;scan</code> requires associativity of the binary operation to parallelize the prefix sum.

Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>inclusive&#95;scan</code> to compute an in-place prefix sum using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
...

int data[6] = {1, 0, 2, 2, 1, 3};

thrust::inclusive_scan(thrust::host, data, data + 6, data); // in-place scan

// data is now {1, 1, 3, 5, 6, 9}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>, and if <code>x</code> and <code>y</code> are objects of <code>OutputIterator's</code><code>value&#95;type</code>, then <code>x + y</code> is defined. If <code>T</code> is <code>OutputIterator's</code><code>value&#95;type</code>, then <code>T(0)</code> is defined.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the input sequence. 
* **`last`** The end of the input sequence. 
* **`result`** The beginning of the output sequence. 

**Preconditions**:
<code>first</code> may equal <code>result</code> but the range <code>[first, last)</code> and the range <code>[result, result + (last - first))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/partial_sum">https://en.cppreference.com/w/cpp/algorithm/partial_sum</a>

<h3 id="function-inclusive-scan">
Function <code>thrust::inclusive&#95;scan</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b>inclusive_scan</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>inclusive&#95;scan</code> computes an inclusive prefix sum operation. The term 'inclusive' means that each result includes the corresponding input operand in the partial sum. More precisely, <code>&#42;first</code> is assigned to <code>&#42;result</code> and the sum of <code>&#42;first</code> and <code>&#42;(first + 1)</code> is assigned to <code>&#42;(result + 1)</code>, and so on. This version of <code>inclusive&#95;scan</code> assumes plus as the associative operator. 

 When the input and output sequences are the same, the scan is performed in-place.

<code>inclusive&#95;scan</code> is similar to <code>std::partial&#95;sum</code> in the STL. The primary difference between the two functions is that <code>std::partial&#95;sum</code> guarantees a serial summation order, while <code>inclusive&#95;scan</code> requires associativity of the binary operation to parallelize the prefix sum.

Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run.


The following code snippet demonstrates how to use <code>inclusive&#95;scan</code>



```cpp
#include <thrust/scan.h>

int data[6] = {1, 0, 2, 2, 1, 3};

thrust::inclusive_scan(data, data + 6, data); // in-place scan

// data is now {1, 1, 3, 5, 6, 9}
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>, and if <code>x</code> and <code>y</code> are objects of <code>OutputIterator's</code><code>value&#95;type</code>, then <code>x + y</code> is defined. If <code>T</code> is <code>OutputIterator's</code><code>value&#95;type</code>, then <code>T(0)</code> is defined.

**Function Parameters**:
* **`first`** The beginning of the input sequence. 
* **`last`** The end of the input sequence. 
* **`result`** The beginning of the output sequence. 

**Preconditions**:
<code>first</code> may equal <code>result</code> but the range <code>[first, last)</code> and the range <code>[result, result + (last - first))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/partial_sum">https://en.cppreference.com/w/cpp/algorithm/partial_sum</a>

<h3 id="function-inclusive-scan">
Function <code>thrust::inclusive&#95;scan</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename AssociativeOperator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>inclusive_scan</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;AssociativeOperator binary_op);</span></code>
<code>inclusive&#95;scan</code> computes an inclusive prefix sum operation. The term 'inclusive' means that each result includes the corresponding input operand in the partial sum. When the input and output sequences are the same, the scan is performed in-place.

<code>inclusive&#95;scan</code> is similar to <code>std::partial&#95;sum</code> in the STL. The primary difference between the two functions is that <code>std::partial&#95;sum</code> guarantees a serial summation order, while <code>inclusive&#95;scan</code> requires associativity of the binary operation to parallelize the prefix sum.

Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>inclusive&#95;scan</code> to compute an in-place prefix sum using the <code>thrust::host</code> execution policy for parallelization:



```cpp
int data[10] = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};

thrust::maximum<int> binary_op;

thrust::inclusive_scan(thrust::host, data, data + 10, data, binary_op); // in-place scan

// data is now {-5, 0, 2, 2, 2, 4, 4, 4, 4, 8}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and <code>OutputIterator's</code><code>value&#95;type</code> is convertible to both <code>AssociativeOperator's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>. 
* **`AssociativeOperator`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a> and <code>AssociativeOperator's</code><code>result&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the input sequence. 
* **`last`** The end of the input sequence. 
* **`result`** The beginning of the output sequence. 
* **`binary_op`** The associatve operator used to 'sum' values. 

**Preconditions**:
<code>first</code> may equal <code>result</code> but the range <code>[first, last)</code> and the range <code>[result, result + (last - first))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/partial_sum">https://en.cppreference.com/w/cpp/algorithm/partial_sum</a>

<h3 id="function-inclusive-scan">
Function <code>thrust::inclusive&#95;scan</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename AssociativeOperator&gt;</span>
<span>OutputIterator </span><span><b>inclusive_scan</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;AssociativeOperator binary_op);</span></code>
<code>inclusive&#95;scan</code> computes an inclusive prefix sum operation. The term 'inclusive' means that each result includes the corresponding input operand in the partial sum. When the input and output sequences are the same, the scan is performed in-place.

<code>inclusive&#95;scan</code> is similar to <code>std::partial&#95;sum</code> in the STL. The primary difference between the two functions is that <code>std::partial&#95;sum</code> guarantees a serial summation order, while <code>inclusive&#95;scan</code> requires associativity of the binary operation to parallelize the prefix sum.

Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run.


The following code snippet demonstrates how to use <code>inclusive&#95;scan</code>



```cpp
int data[10] = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};

thrust::maximum<int> binary_op;

thrust::inclusive_scan(data, data + 10, data, binary_op); // in-place scan

// data is now {-5, 0, 2, 2, 2, 4, 4, 4, 4, 8}
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and <code>OutputIterator's</code><code>value&#95;type</code> is convertible to both <code>AssociativeOperator's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>. 
* **`AssociativeOperator`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a> and <code>AssociativeOperator's</code><code>result&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The beginning of the input sequence. 
* **`last`** The end of the input sequence. 
* **`result`** The beginning of the output sequence. 
* **`binary_op`** The associatve operator used to 'sum' values. 

**Preconditions**:
<code>first</code> may equal <code>result</code> but the range <code>[first, last)</code> and the range <code>[result, result + (last - first))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/partial_sum">https://en.cppreference.com/w/cpp/algorithm/partial_sum</a>

<h3 id="function-exclusive-scan">
Function <code>thrust::exclusive&#95;scan</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>exclusive_scan</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>exclusive&#95;scan</code> computes an exclusive prefix sum operation. The term 'exclusive' means that each result does not include the corresponding input operand in the partial sum. More precisely, <code>0</code> is assigned to <code>&#42;result</code> and the sum of <code>0</code> and <code>&#42;first</code> is assigned to <code>&#42;(result + 1)</code>, and so on. This version of <code>exclusive&#95;scan</code> assumes plus as the associative operator and <code>0</code> as the initial value. When the input and output sequences are the same, the scan is performed in-place.

Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>exclusive&#95;scan</code> to compute an in-place prefix sum using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
...

int data[6] = {1, 0, 2, 2, 1, 3};

thrust::exclusive_scan(thrust::host, data, data + 6, data); // in-place scan

// data is now {0, 1, 1, 3, 5, 6}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>, and if <code>x</code> and <code>y</code> are objects of <code>OutputIterator's</code><code>value&#95;type</code>, then <code>x + y</code> is defined. If <code>T</code> is <code>OutputIterator's</code><code>value&#95;type</code>, then <code>T(0)</code> is defined.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the input sequence. 
* **`last`** The end of the input sequence. 
* **`result`** The beginning of the output sequence. 

**Preconditions**:
<code>first</code> may equal <code>result</code> but the range <code>[first, last)</code> and the range <code>[result, result + (last - first))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/partial_sum">https://en.cppreference.com/w/cpp/algorithm/partial_sum</a>

<h3 id="function-exclusive-scan">
Function <code>thrust::exclusive&#95;scan</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b>exclusive_scan</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>exclusive&#95;scan</code> computes an exclusive prefix sum operation. The term 'exclusive' means that each result does not include the corresponding input operand in the partial sum. More precisely, <code>0</code> is assigned to <code>&#42;result</code> and the sum of <code>0</code> and <code>&#42;first</code> is assigned to <code>&#42;(result + 1)</code>, and so on. This version of <code>exclusive&#95;scan</code> assumes plus as the associative operator and <code>0</code> as the initial value. When the input and output sequences are the same, the scan is performed in-place.

Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run.


The following code snippet demonstrates how to use <code>exclusive&#95;scan</code>



```cpp
#include <thrust/scan.h>

int data[6] = {1, 0, 2, 2, 1, 3};

thrust::exclusive_scan(data, data + 6, data); // in-place scan

// data is now {0, 1, 1, 3, 5, 6}
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>, and if <code>x</code> and <code>y</code> are objects of <code>OutputIterator's</code><code>value&#95;type</code>, then <code>x + y</code> is defined. If <code>T</code> is <code>OutputIterator's</code><code>value&#95;type</code>, then <code>T(0)</code> is defined.

**Function Parameters**:
* **`first`** The beginning of the input sequence. 
* **`last`** The end of the input sequence. 
* **`result`** The beginning of the output sequence. 

**Preconditions**:
<code>first</code> may equal <code>result</code> but the range <code>[first, last)</code> and the range <code>[result, result + (last - first))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/partial_sum">https://en.cppreference.com/w/cpp/algorithm/partial_sum</a>

<h3 id="function-exclusive-scan">
Function <code>thrust::exclusive&#95;scan</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>exclusive_scan</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;T init);</span></code>
<code>exclusive&#95;scan</code> computes an exclusive prefix sum operation. The term 'exclusive' means that each result does not include the corresponding input operand in the partial sum. More precisely, <code>init</code> is assigned to <code>&#42;result</code> and the sum of <code>init</code> and <code>&#42;first</code> is assigned to <code>&#42;(result + 1)</code>, and so on. This version of <code>exclusive&#95;scan</code> assumes plus as the associative operator but requires an initial value <code>init</code>. When the input and output sequences are the same, the scan is performed in-place.

Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>exclusive&#95;scan</code> to compute an in-place prefix sum using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

int data[6] = {1, 0, 2, 2, 1, 3};

thrust::exclusive_scan(thrust::host, data, data + 6, data, 4); // in-place scan

// data is now {4, 5, 5, 7, 9, 10}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>, and if <code>x</code> and <code>y</code> are objects of <code>OutputIterator's</code><code>value&#95;type</code>, then <code>x + y</code> is defined. 
* **`T`** is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the input sequence. 
* **`last`** The end of the input sequence. 
* **`result`** The beginning of the output sequence. 
* **`init`** The initial value. 

**Preconditions**:
<code>first</code> may equal <code>result</code> but the range <code>[first, last)</code> and the range <code>[result, result + (last - first))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/partial_sum">https://en.cppreference.com/w/cpp/algorithm/partial_sum</a>

<h3 id="function-exclusive-scan">
Function <code>thrust::exclusive&#95;scan</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>OutputIterator </span><span><b>exclusive_scan</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;T init);</span></code>
<code>exclusive&#95;scan</code> computes an exclusive prefix sum operation. The term 'exclusive' means that each result does not include the corresponding input operand in the partial sum. More precisely, <code>init</code> is assigned to <code>&#42;result</code> and the sum of <code>init</code> and <code>&#42;first</code> is assigned to <code>&#42;(result + 1)</code>, and so on. This version of <code>exclusive&#95;scan</code> assumes plus as the associative operator but requires an initial value <code>init</code>. When the input and output sequences are the same, the scan is performed in-place.

Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run.


The following code snippet demonstrates how to use <code>exclusive&#95;scan</code>



```cpp
#include <thrust/scan.h>

int data[6] = {1, 0, 2, 2, 1, 3};

thrust::exclusive_scan(data, data + 6, data, 4); // in-place scan

// data is now {4, 5, 5, 7, 9, 10}
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>, and if <code>x</code> and <code>y</code> are objects of <code>OutputIterator's</code><code>value&#95;type</code>, then <code>x + y</code> is defined. 
* **`T`** is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The beginning of the input sequence. 
* **`last`** The end of the input sequence. 
* **`result`** The beginning of the output sequence. 
* **`init`** The initial value. 

**Preconditions**:
<code>first</code> may equal <code>result</code> but the range <code>[first, last)</code> and the range <code>[result, result + (last - first))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/partial_sum">https://en.cppreference.com/w/cpp/algorithm/partial_sum</a>

<h3 id="function-exclusive-scan">
Function <code>thrust::exclusive&#95;scan</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename AssociativeOperator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>exclusive_scan</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;T init,</span>
<span>&nbsp;&nbsp;AssociativeOperator binary_op);</span></code>
<code>exclusive&#95;scan</code> computes an exclusive prefix sum operation. The term 'exclusive' means that each result does not include the corresponding input operand in the partial sum. More precisely, <code>init</code> is assigned to <code>&#42;result</code> and the value <code>binary&#95;op(init, &#42;first)</code> is assigned to <code>&#42;(result + 1)</code>, and so on. This version of the function requires both an associative operator and an initial value <code>init</code>. When the input and output sequences are the same, the scan is performed in-place.

Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>exclusive&#95;scan</code> to compute an in-place prefix sum using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...

int data[10] = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};

thrust::maximum<int> binary_op;

thrust::exclusive_scan(thrust::host, data, data + 10, data, 1, binary_op); // in-place scan

// data is now {1, 1, 1, 2, 2, 2, 4, 4, 4, 4 }
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and <code>OutputIterator's</code><code>value&#95;type</code> is convertible to both <code>AssociativeOperator's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>. 
* **`T`** is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`AssociativeOperator`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a> and <code>AssociativeOperator's</code><code>result&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the input sequence. 
* **`last`** The end of the input sequence. 
* **`result`** The beginning of the output sequence. 
* **`init`** The initial value. 
* **`binary_op`** The associatve operator used to 'sum' values. 

**Preconditions**:
<code>first</code> may equal <code>result</code> but the range <code>[first, last)</code> and the range <code>[result, result + (last - first))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/partial_sum">https://en.cppreference.com/w/cpp/algorithm/partial_sum</a>

<h3 id="function-exclusive-scan">
Function <code>thrust::exclusive&#95;scan</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename AssociativeOperator&gt;</span>
<span>OutputIterator </span><span><b>exclusive_scan</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;T init,</span>
<span>&nbsp;&nbsp;AssociativeOperator binary_op);</span></code>
<code>exclusive&#95;scan</code> computes an exclusive prefix sum operation. The term 'exclusive' means that each result does not include the corresponding input operand in the partial sum. More precisely, <code>init</code> is assigned to <code>&#42;result</code> and the value <code>binary&#95;op(init, &#42;first)</code> is assigned to <code>&#42;(result + 1)</code>, and so on. This version of the function requires both an associative operator and an initial value <code>init</code>. When the input and output sequences are the same, the scan is performed in-place.

Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run.


The following code snippet demonstrates how to use <code>exclusive&#95;scan</code>



```cpp
#include <thrust/scan.h>
#include <thrust/functional.h>

int data[10] = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};

thrust::maximum<int> binary_op;

thrust::exclusive_scan(data, data + 10, data, 1, binary_op); // in-place scan

// data is now {1, 1, 1, 2, 2, 2, 4, 4, 4, 4 }
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and <code>OutputIterator's</code><code>value&#95;type</code> is convertible to both <code>AssociativeOperator's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>. 
* **`T`** is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`AssociativeOperator`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a> and <code>AssociativeOperator's</code><code>result&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The beginning of the input sequence. 
* **`last`** The end of the input sequence. 
* **`result`** The beginning of the output sequence. 
* **`init`** The initial value. 
* **`binary_op`** The associatve operator used to 'sum' values. 

**Preconditions**:
<code>first</code> may equal <code>result</code> but the range <code>[first, last)</code> and the range <code>[result, result + (last - first))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/partial_sum">https://en.cppreference.com/w/cpp/algorithm/partial_sum</a>


