---
title: Transformed Prefix Sums
parent: Prefix Sums
grand_parent: Algorithms
nav_exclude: false
has_children: true
has_toc: false
---

# Transformed Prefix Sums

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction,</span>
<span>&nbsp;&nbsp;typename AssociativeOperator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformed__prefixsums.html#function-transform-inclusive-scan">thrust::transform&#95;inclusive&#95;scan</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;UnaryFunction unary_op,</span>
<span>&nbsp;&nbsp;AssociativeOperator binary_op);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction,</span>
<span>&nbsp;&nbsp;typename AssociativeOperator&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformed__prefixsums.html#function-transform-inclusive-scan">thrust::transform&#95;inclusive&#95;scan</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;UnaryFunction unary_op,</span>
<span>&nbsp;&nbsp;AssociativeOperator binary_op);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename AssociativeOperator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformed__prefixsums.html#function-transform-exclusive-scan">thrust::transform&#95;exclusive&#95;scan</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;UnaryFunction unary_op,</span>
<span>&nbsp;&nbsp;T init,</span>
<span>&nbsp;&nbsp;AssociativeOperator binary_op);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename AssociativeOperator&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__transformed__prefixsums.html#function-transform-exclusive-scan">thrust::transform&#95;exclusive&#95;scan</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;UnaryFunction unary_op,</span>
<span>&nbsp;&nbsp;T init,</span>
<span>&nbsp;&nbsp;AssociativeOperator binary_op);</span>
</code>

## Functions

<h3 id="function-transform-inclusive-scan">
Function <code>thrust::transform&#95;inclusive&#95;scan</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction,</span>
<span>&nbsp;&nbsp;typename AssociativeOperator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>transform_inclusive_scan</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;UnaryFunction unary_op,</span>
<span>&nbsp;&nbsp;AssociativeOperator binary_op);</span></code>
<code>transform&#95;inclusive&#95;scan</code> fuses the <code>transform</code> and <code>inclusive&#95;scan</code> operations. <code>transform&#95;inclusive&#95;scan</code> is equivalent to performing a tranformation defined by <code>unary&#95;op</code> into a temporary sequence and then performing an <code>inclusive&#95;scan</code> on the tranformed sequence. In most cases, fusing these two operations together is more efficient, since fewer memory reads and writes are required. In <code>transform&#95;inclusive&#95;scan</code>, <code>unary&#95;op(&#42;first)</code> is assigned to <code>&#42;result</code> and the result of <code>binary&#95;op(unary&#95;op(&#42;first), unary&#95;op(&#42;(first + 1)))</code> is assigned to <code>&#42;(result + 1)</code>, and so on. The transform scan operation is permitted to be in-place.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>transform&#95;inclusive&#95;scan</code> using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/transform_scan.h>
#include <thrust/execution_policy.h>
...

int data[6] = {1, 0, 2, 2, 1, 3};

thrust::negate<int> unary_op;
thrust::plus<int> binary_op;

thrust::transform_inclusive_scan(thrust::host, data, data + 6, data, unary_op, binary_op); // in-place scan

// data is now {-1, -1, -3, -5, -6, -9}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>unary&#95;op's</code> input type. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`UnaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a> and accepts inputs of <code>InputIterator's</code><code>value&#95;type</code>. <code>UnaryFunction's</code> result_type is convertable to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`AssociativeOperator`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a> and <code>AssociativeOperator's</code><code>result&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the input sequence. 
* **`last`** The end of the input sequence. 
* **`result`** The beginning of the output sequence. 
* **`unary_op`** The function used to tranform the input sequence. 
* **`binary_op`** The associatve operator used to 'sum' transformed values. 

**Preconditions**:
<code>first</code> may equal <code>result</code>, but the range <code>[first, last)</code> and the range <code>[result, result + (last - first))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
* <code>transform</code>
* <code>inclusive&#95;scan</code>

<h3 id="function-transform-inclusive-scan">
Function <code>thrust::transform&#95;inclusive&#95;scan</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction,</span>
<span>&nbsp;&nbsp;typename AssociativeOperator&gt;</span>
<span>OutputIterator </span><span><b>transform_inclusive_scan</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;UnaryFunction unary_op,</span>
<span>&nbsp;&nbsp;AssociativeOperator binary_op);</span></code>
<code>transform&#95;inclusive&#95;scan</code> fuses the <code>transform</code> and <code>inclusive&#95;scan</code> operations. <code>transform&#95;inclusive&#95;scan</code> is equivalent to performing a tranformation defined by <code>unary&#95;op</code> into a temporary sequence and then performing an <code>inclusive&#95;scan</code> on the tranformed sequence. In most cases, fusing these two operations together is more efficient, since fewer memory reads and writes are required. In <code>transform&#95;inclusive&#95;scan</code>, <code>unary&#95;op(&#42;first)</code> is assigned to <code>&#42;result</code> and the result of <code>binary&#95;op(unary&#95;op(&#42;first), unary&#95;op(&#42;(first + 1)))</code> is assigned to <code>&#42;(result + 1)</code>, and so on. The transform scan operation is permitted to be in-place.


The following code snippet demonstrates how to use <code>transform&#95;inclusive&#95;scan</code>



```cpp
#include <thrust/transform_scan.h>

int data[6] = {1, 0, 2, 2, 1, 3};

thrust::negate<int> unary_op;
thrust::plus<int> binary_op;

thrust::transform_inclusive_scan(data, data + 6, data, unary_op, binary_op); // in-place scan

// data is now {-1, -1, -3, -5, -6, -9}
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>unary&#95;op's</code> input type. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`UnaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a> and accepts inputs of <code>InputIterator's</code><code>value&#95;type</code>. <code>UnaryFunction's</code> result_type is convertable to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`AssociativeOperator`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a> and <code>AssociativeOperator's</code><code>result&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The beginning of the input sequence. 
* **`last`** The end of the input sequence. 
* **`result`** The beginning of the output sequence. 
* **`unary_op`** The function used to tranform the input sequence. 
* **`binary_op`** The associatve operator used to 'sum' transformed values. 

**Preconditions**:
<code>first</code> may equal <code>result</code>, but the range <code>[first, last)</code> and the range <code>[result, result + (last - first))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
* <code>transform</code>
* <code>inclusive&#95;scan</code>

<h3 id="function-transform-exclusive-scan">
Function <code>thrust::transform&#95;exclusive&#95;scan</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename AssociativeOperator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>transform_exclusive_scan</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;UnaryFunction unary_op,</span>
<span>&nbsp;&nbsp;T init,</span>
<span>&nbsp;&nbsp;AssociativeOperator binary_op);</span></code>
<code>transform&#95;exclusive&#95;scan</code> fuses the <code>transform</code> and <code>exclusive&#95;scan</code> operations. <code>transform&#95;exclusive&#95;scan</code> is equivalent to performing a tranformation defined by <code>unary&#95;op</code> into a temporary sequence and then performing an <code>exclusive&#95;scan</code> on the tranformed sequence. In most cases, fusing these two operations together is more efficient, since fewer memory reads and writes are required. In <code>transform&#95;exclusive&#95;scan</code>, <code>init</code> is assigned to <code>&#42;result</code> and the result of <code>binary&#95;op(init, unary&#95;op(&#42;first))</code> is assigned to <code>&#42;(result + 1)</code>, and so on. The transform scan operation is permitted to be in-place.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>transform&#95;exclusive&#95;scan</code> using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/transform_scan.h>
#include <thrust/execution_policy.h>
...

int data[6] = {1, 0, 2, 2, 1, 3};

thrust::negate<int> unary_op;
thrust::plus<int> binary_op;

thrust::transform_exclusive_scan(thrust::host, data, data + 6, data, unary_op, 4, binary_op); // in-place scan

// data is now {4, 3, 3, 1, -1, -2}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>unary&#95;op's</code> input type. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`UnaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a> and accepts inputs of <code>InputIterator's</code><code>value&#95;type</code>. <code>UnaryFunction's</code> result_type is convertable to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`T`** is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`AssociativeOperator`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a> and <code>AssociativeOperator's</code><code>result&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the input sequence. 
* **`last`** The end of the input sequence. 
* **`result`** The beginning of the output sequence. 
* **`unary_op`** The function used to tranform the input sequence. 
* **`init`** The initial value of the <code>exclusive&#95;scan</code>
* **`binary_op`** The associatve operator used to 'sum' transformed values. 

**Preconditions**:
<code>first</code> may equal <code>result</code>, but the range <code>[first, last)</code> and the range <code>[result, result + (last - first))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
* <code>transform</code>
* <code>exclusive&#95;scan</code>

<h3 id="function-transform-exclusive-scan">
Function <code>thrust::transform&#95;exclusive&#95;scan</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename UnaryFunction,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename AssociativeOperator&gt;</span>
<span>OutputIterator </span><span><b>transform_exclusive_scan</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;UnaryFunction unary_op,</span>
<span>&nbsp;&nbsp;T init,</span>
<span>&nbsp;&nbsp;AssociativeOperator binary_op);</span></code>
<code>transform&#95;exclusive&#95;scan</code> fuses the <code>transform</code> and <code>exclusive&#95;scan</code> operations. <code>transform&#95;exclusive&#95;scan</code> is equivalent to performing a tranformation defined by <code>unary&#95;op</code> into a temporary sequence and then performing an <code>exclusive&#95;scan</code> on the tranformed sequence. In most cases, fusing these two operations together is more efficient, since fewer memory reads and writes are required. In <code>transform&#95;exclusive&#95;scan</code>, <code>init</code> is assigned to <code>&#42;result</code> and the result of <code>binary&#95;op(init, unary&#95;op(&#42;first))</code> is assigned to <code>&#42;(result + 1)</code>, and so on. The transform scan operation is permitted to be in-place.


The following code snippet demonstrates how to use <code>transform&#95;exclusive&#95;scan</code>



```cpp
#include <thrust/transform_scan.h>

int data[6] = {1, 0, 2, 2, 1, 3};

thrust::negate<int> unary_op;
thrust::plus<int> binary_op;

thrust::transform_exclusive_scan(data, data + 6, data, unary_op, 4, binary_op); // in-place scan

// data is now {4, 3, 3, 1, -1, -2}
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>unary&#95;op's</code> input type. 
* **`OutputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`UnaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a> and accepts inputs of <code>InputIterator's</code><code>value&#95;type</code>. <code>UnaryFunction's</code> result_type is convertable to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`T`** is convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`AssociativeOperator`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a> and <code>AssociativeOperator's</code><code>result&#95;type</code> is convertible to <code>OutputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The beginning of the input sequence. 
* **`last`** The end of the input sequence. 
* **`result`** The beginning of the output sequence. 
* **`unary_op`** The function used to tranform the input sequence. 
* **`init`** The initial value of the <code>exclusive&#95;scan</code>
* **`binary_op`** The associatve operator used to 'sum' transformed values. 

**Preconditions**:
<code>first</code> may equal <code>result</code>, but the range <code>[first, last)</code> and the range <code>[result, result + (last - first))</code> shall not overlap otherwise.

**Returns**:
The end of the output sequence.

**See**:
* <code>transform</code>
* <code>exclusive&#95;scan</code>


