---
title: Reductions
parent: Algorithms
grand_parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Reductions

## Groups

* **[Comparisons]({{ site.baseurl }}/api/groups/group__comparisons.html)**
* **[Counting]({{ site.baseurl }}/api/groups/group__counting.html)**
* **[Extrema]({{ site.baseurl }}/api/groups/group__extrema.html)**
* **[Logical]({{ site.baseurl }}/api/groups/group__logical.html)**
* **[Predicates]({{ site.baseurl }}/api/groups/group__predicates.html)**
* **[Transformed Reductions]({{ site.baseurl }}/api/groups/group__transformed__reductions.html)**

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">thrust::iterator_traits</a>< InputIterator >::value_type </span><span><b><a href="{{ site.baseurl }}/api/groups/group__reductions.html#function-reduce">thrust::reduce</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last);</span>
<br>
<span>template &lt;typename InputIterator&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">thrust::iterator_traits</a>< InputIterator >::value_type </span><span><b><a href="{{ site.baseurl }}/api/groups/group__reductions.html#function-reduce">thrust::reduce</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ T </span><span><b><a href="{{ site.baseurl }}/api/groups/group__reductions.html#function-reduce">thrust::reduce</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;T init);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>T </span><span><b><a href="{{ site.baseurl }}/api/groups/group__reductions.html#function-reduce">thrust::reduce</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;T init);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename BinaryFunction&gt;</span>
<span>__host__ __device__ T </span><span><b><a href="{{ site.baseurl }}/api/groups/group__reductions.html#function-reduce">thrust::reduce</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;T init,</span>
<span>&nbsp;&nbsp;BinaryFunction binary_op);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename BinaryFunction&gt;</span>
<span>T </span><span><b><a href="{{ site.baseurl }}/api/groups/group__reductions.html#function-reduce">thrust::reduce</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;T init,</span>
<span>&nbsp;&nbsp;BinaryFunction binary_op);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__reductions.html#function-reduce-by-key">thrust::reduce&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last,</span>
<span>&nbsp;&nbsp;InputIterator2 values_first,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_output,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_output);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__reductions.html#function-reduce-by-key">thrust::reduce&#95;by&#95;key</a></b>(InputIterator1 keys_first,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last,</span>
<span>&nbsp;&nbsp;InputIterator2 values_first,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_output,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_output);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__reductions.html#function-reduce-by-key">thrust::reduce&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last,</span>
<span>&nbsp;&nbsp;InputIterator2 values_first,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_output,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_output,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__reductions.html#function-reduce-by-key">thrust::reduce&#95;by&#95;key</a></b>(InputIterator1 keys_first,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last,</span>
<span>&nbsp;&nbsp;InputIterator2 values_first,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_output,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_output,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate,</span>
<span>&nbsp;&nbsp;typename BinaryFunction&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__reductions.html#function-reduce-by-key">thrust::reduce&#95;by&#95;key</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last,</span>
<span>&nbsp;&nbsp;InputIterator2 values_first,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_output,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_output,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred,</span>
<span>&nbsp;&nbsp;BinaryFunction binary_op);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate,</span>
<span>&nbsp;&nbsp;typename BinaryFunction&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__reductions.html#function-reduce-by-key">thrust::reduce&#95;by&#95;key</a></b>(InputIterator1 keys_first,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last,</span>
<span>&nbsp;&nbsp;InputIterator2 values_first,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_output,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_output,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred,</span>
<span>&nbsp;&nbsp;BinaryFunction binary_op);</span>
</code>

## Functions

<h3 id="function-reduce">
Function <code>thrust::reduce</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">thrust::iterator_traits</a>< InputIterator >::value_type </span><span><b>reduce</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last);</span></code>
<code>reduce</code> is a generalization of summation: it computes the sum (or some other binary operation) of all the elements in the range <code>[first, last)</code>. This version of <code>reduce</code> uses <code>0</code> as the initial value of the reduction. <code>reduce</code> is similar to the C++ Standard Template Library's <code>std::accumulate</code>. The primary difference between the two functions is that <code>std::accumulate</code> guarantees the order of summation, while <code>reduce</code> requires associativity of the binary operation to parallelize the reduction.

Note that <code>reduce</code> also assumes that the binary reduction operator (in this case operator+) is commutative. If the reduction operator is not commutative then <code>thrust::reduce</code> should not be used. Instead, one could use <code>inclusive&#95;scan</code> (which does not require commutativity) and select the last element of the output array.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>reduce</code> to compute the sum of a sequence of integers using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
...
int data[6] = {1, 0, 2, 2, 1, 3};
int result = thrust::reduce(thrust::host, data, data + 6);

// result == 9
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and if <code>x</code> and <code>y</code> are objects of <code>InputIterator's</code><code>value&#95;type</code>, then <code>x + y</code> is defined and is convertible to <code>InputIterator's</code><code>value&#95;type</code>. If <code>T</code> is <code>InputIterator's</code><code>value&#95;type</code>, then <code>T(0)</code> is defined.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 

**Returns**:
The result of the reduction.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/accumulate">https://en.cppreference.com/w/cpp/algorithm/accumulate</a>

<h3 id="function-reduce">
Function <code>thrust::reduce</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">thrust::iterator_traits</a>< InputIterator >::value_type </span><span><b>reduce</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last);</span></code>
<code>reduce</code> is a generalization of summation: it computes the sum (or some other binary operation) of all the elements in the range <code>[first, last)</code>. This version of <code>reduce</code> uses <code>0</code> as the initial value of the reduction. <code>reduce</code> is similar to the C++ Standard Template Library's <code>std::accumulate</code>. The primary difference between the two functions is that <code>std::accumulate</code> guarantees the order of summation, while <code>reduce</code> requires associativity of the binary operation to parallelize the reduction.

Note that <code>reduce</code> also assumes that the binary reduction operator (in this case operator+) is commutative. If the reduction operator is not commutative then <code>thrust::reduce</code> should not be used. Instead, one could use <code>inclusive&#95;scan</code> (which does not require commutativity) and select the last element of the output array.


The following code snippet demonstrates how to use <code>reduce</code> to compute the sum of a sequence of integers.



```cpp
#include <thrust/reduce.h>
...
int data[6] = {1, 0, 2, 2, 1, 3};
int result = thrust::reduce(data, data + 6);

// result == 9
```

**Template Parameters**:
**`InputIterator`**: is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and if <code>x</code> and <code>y</code> are objects of <code>InputIterator's</code><code>value&#95;type</code>, then <code>x + y</code> is defined and is convertible to <code>InputIterator's</code><code>value&#95;type</code>. If <code>T</code> is <code>InputIterator's</code><code>value&#95;type</code>, then <code>T(0)</code> is defined.

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 

**Returns**:
The result of the reduction.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/accumulate">https://en.cppreference.com/w/cpp/algorithm/accumulate</a>

<h3 id="function-reduce">
Function <code>thrust::reduce</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>__host__ __device__ T </span><span><b>reduce</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;T init);</span></code>
<code>reduce</code> is a generalization of summation: it computes the sum (or some other binary operation) of all the elements in the range <code>[first, last)</code>. This version of <code>reduce</code> uses <code>init</code> as the initial value of the reduction. <code>reduce</code> is similar to the C++ Standard Template Library's <code>std::accumulate</code>. The primary difference between the two functions is that <code>std::accumulate</code> guarantees the order of summation, while <code>reduce</code> requires associativity of the binary operation to parallelize the reduction.

Note that <code>reduce</code> also assumes that the binary reduction operator (in this case operator+) is commutative. If the reduction operator is not commutative then <code>thrust::reduce</code> should not be used. Instead, one could use <code>inclusive&#95;scan</code> (which does not require commutativity) and select the last element of the output array.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>reduce</code> to compute the sum of a sequence of integers including an intialization value using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
...
int data[6] = {1, 0, 2, 2, 1, 3};
int result = thrust::reduce(thrust::host, data, data + 6, 1);

// result == 10
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and if <code>x</code> and <code>y</code> are objects of <code>InputIterator's</code><code>value&#95;type</code>, then <code>x + y</code> is defined and is convertible to <code>T</code>. 
* **`T`** is convertible to <code>InputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the input sequence. 
* **`last`** The end of the input sequence. 
* **`init`** The initial value. 

**Returns**:
The result of the reduction.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/accumulate">https://en.cppreference.com/w/cpp/algorithm/accumulate</a>

<h3 id="function-reduce">
Function <code>thrust::reduce</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>T </span><span><b>reduce</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;T init);</span></code>
<code>reduce</code> is a generalization of summation: it computes the sum (or some other binary operation) of all the elements in the range <code>[first, last)</code>. This version of <code>reduce</code> uses <code>init</code> as the initial value of the reduction. <code>reduce</code> is similar to the C++ Standard Template Library's <code>std::accumulate</code>. The primary difference between the two functions is that <code>std::accumulate</code> guarantees the order of summation, while <code>reduce</code> requires associativity of the binary operation to parallelize the reduction.

Note that <code>reduce</code> also assumes that the binary reduction operator (in this case operator+) is commutative. If the reduction operator is not commutative then <code>thrust::reduce</code> should not be used. Instead, one could use <code>inclusive&#95;scan</code> (which does not require commutativity) and select the last element of the output array.


The following code snippet demonstrates how to use <code>reduce</code> to compute the sum of a sequence of integers including an intialization value.



```cpp
#include <thrust/reduce.h>
...
int data[6] = {1, 0, 2, 2, 1, 3};
int result = thrust::reduce(data, data + 6, 1);

// result == 10
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and if <code>x</code> and <code>y</code> are objects of <code>InputIterator's</code><code>value&#95;type</code>, then <code>x + y</code> is defined and is convertible to <code>T</code>. 
* **`T`** is convertible to <code>InputIterator's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`first`** The beginning of the input sequence. 
* **`last`** The end of the input sequence. 
* **`init`** The initial value. 

**Returns**:
The result of the reduction.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/accumulate">https://en.cppreference.com/w/cpp/algorithm/accumulate</a>

<h3 id="function-reduce">
Function <code>thrust::reduce</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename BinaryFunction&gt;</span>
<span>__host__ __device__ T </span><span><b>reduce</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;T init,</span>
<span>&nbsp;&nbsp;BinaryFunction binary_op);</span></code>
<code>reduce</code> is a generalization of summation: it computes the sum (or some other binary operation) of all the elements in the range <code>[first, last)</code>. This version of <code>reduce</code> uses <code>init</code> as the initial value of the reduction and <code>binary&#95;op</code> as the binary function used for summation. <code>reduce</code> is similar to the C++ Standard Template Library's <code>std::accumulate</code>. The primary difference between the two functions is that <code>std::accumulate</code> guarantees the order of summation, while <code>reduce</code> requires associativity of <code>binary&#95;op</code> to parallelize the reduction.

Note that <code>reduce</code> also assumes that the binary reduction operator (in this case <code>binary&#95;op</code>) is commutative. If the reduction operator is not commutative then <code>thrust::reduce</code> should not be used. Instead, one could use <code>inclusive&#95;scan</code> (which does not require commutativity) and select the last element of the output array.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>reduce</code> to compute the maximum value of a sequence of integers using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...
int data[6] = {1, 0, 2, 2, 1, 3};
int result = thrust::reduce(thrust::host,
                            data, data + 6,
                            -1,
                            thrust::maximum<int>());
// result == 3
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>T</code>. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and is convertible to <code>BinaryFunction's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>. 
* **`BinaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>, and <code>BinaryFunction's</code><code>result&#95;type</code> is convertible to <code>OutputType</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the input sequence. 
* **`last`** The end of the input sequence. 
* **`init`** The initial value. 
* **`binary_op`** The binary function used to 'sum' values. 

**Returns**:
The result of the reduction.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/accumulate">https://en.cppreference.com/w/cpp/algorithm/accumulate</a>
* transform_reduce 

<h3 id="function-reduce">
Function <code>thrust::reduce</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename T,</span>
<span>&nbsp;&nbsp;typename BinaryFunction&gt;</span>
<span>T </span><span><b>reduce</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;T init,</span>
<span>&nbsp;&nbsp;BinaryFunction binary_op);</span></code>
<code>reduce</code> is a generalization of summation: it computes the sum (or some other binary operation) of all the elements in the range <code>[first, last)</code>. This version of <code>reduce</code> uses <code>init</code> as the initial value of the reduction and <code>binary&#95;op</code> as the binary function used for summation. <code>reduce</code> is similar to the C++ Standard Template Library's <code>std::accumulate</code>. The primary difference between the two functions is that <code>std::accumulate</code> guarantees the order of summation, while <code>reduce</code> requires associativity of <code>binary&#95;op</code> to parallelize the reduction.

Note that <code>reduce</code> also assumes that the binary reduction operator (in this case <code>binary&#95;op</code>) is commutative. If the reduction operator is not commutative then <code>thrust::reduce</code> should not be used. Instead, one could use <code>inclusive&#95;scan</code> (which does not require commutativity) and select the last element of the output array.


The following code snippet demonstrates how to use <code>reduce</code> to compute the maximum value of a sequence of integers.



```cpp
#include <thrust/reduce.h>
#include <thrust/functional.h>
...
int data[6] = {1, 0, 2, 2, 1, 3};
int result = thrust::reduce(data, data + 6,
                            -1,
                            thrust::maximum<int>());
// result == 3
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>T</code>. 
* **`T`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and is convertible to <code>BinaryFunction's</code><code>first&#95;argument&#95;type</code> and <code>second&#95;argument&#95;type</code>. 
* **`BinaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>, and <code>BinaryFunction's</code><code>result&#95;type</code> is convertible to <code>OutputType</code>.

**Function Parameters**:
* **`first`** The beginning of the input sequence. 
* **`last`** The end of the input sequence. 
* **`init`** The initial value. 
* **`binary_op`** The binary function used to 'sum' values. 

**Returns**:
The result of the reduction.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/accumulate">https://en.cppreference.com/w/cpp/algorithm/accumulate</a>
* transform_reduce 

<h3 id="function-reduce-by-key">
Function <code>thrust::reduce&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>reduce_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last,</span>
<span>&nbsp;&nbsp;InputIterator2 values_first,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_output,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_output);</span></code>
<code>reduce&#95;by&#95;key</code> is a generalization of <code>reduce</code> to key-value pairs. For each group of consecutive keys in the range <code>[keys&#95;first, keys&#95;last)</code> that are equal, <code>reduce&#95;by&#95;key</code> copies the first element of the group to the <code>keys&#95;output</code>. The corresponding values in the range are reduced using the <code>plus</code> and the result copied to <code>values&#95;output</code>.

This version of <code>reduce&#95;by&#95;key</code> uses the function object <code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1equal__to.html">equal&#95;to</a></code> to test for equality and <code>plus</code> to reduce values with equal keys.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>reduce&#95;by&#95;key</code> to compact a sequence of key/value pairs and sum values with equal keys using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
int C[N];                         // output keys
int D[N];                         // output values

thrust::pair<int*,int*> new_end;
new_end = thrust::reduce_by_key(thrust::host, A, A + N, B, C, D);

// The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
// The first four values in D are now {9, 21, 9, 3} and new_end.second - D is 4.
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
* **`keys_output`** The beginning of the output key range. 
* **`values_output`** The beginning of the output value range. 

**Preconditions**:
The input ranges shall not overlap either output range.

**Returns**:
A pair of iterators at end of the ranges <code>[keys&#95;output, keys&#95;output&#95;last)</code> and <code>[values&#95;output, values&#95;output&#95;last)</code>.

**See**:
* reduce 
* unique_copy 
* unique_by_key 
* unique_by_key_copy 

<h3 id="function-reduce-by-key">
Function <code>thrust::reduce&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>reduce_by_key</b>(InputIterator1 keys_first,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last,</span>
<span>&nbsp;&nbsp;InputIterator2 values_first,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_output,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_output);</span></code>
<code>reduce&#95;by&#95;key</code> is a generalization of <code>reduce</code> to key-value pairs. For each group of consecutive keys in the range <code>[keys&#95;first, keys&#95;last)</code> that are equal, <code>reduce&#95;by&#95;key</code> copies the first element of the group to the <code>keys&#95;output</code>. The corresponding values in the range are reduced using the <code>plus</code> and the result copied to <code>values&#95;output</code>.

This version of <code>reduce&#95;by&#95;key</code> uses the function object <code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1equal__to.html">equal&#95;to</a></code> to test for equality and <code>plus</code> to reduce values with equal keys.


The following code snippet demonstrates how to use <code>reduce&#95;by&#95;key</code> to compact a sequence of key/value pairs and sum values with equal keys.



```cpp
#include <thrust/reduce.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
int C[N];                         // output keys
int D[N];                         // output values

thrust::pair<int*,int*> new_end;
new_end = thrust::reduce_by_key(A, A + N, B, C, D);

// The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
// The first four values in D are now {9, 21, 9, 3} and new_end.second - D is 4.
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
* **`keys_output`** The beginning of the output key range. 
* **`values_output`** The beginning of the output value range. 

**Preconditions**:
The input ranges shall not overlap either output range.

**Returns**:
A pair of iterators at end of the ranges <code>[keys&#95;output, keys&#95;output&#95;last)</code> and <code>[values&#95;output, values&#95;output&#95;last)</code>.

**See**:
* reduce 
* unique_copy 
* unique_by_key 
* unique_by_key_copy 

<h3 id="function-reduce-by-key">
Function <code>thrust::reduce&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>reduce_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last,</span>
<span>&nbsp;&nbsp;InputIterator2 values_first,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_output,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_output,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span></code>
<code>reduce&#95;by&#95;key</code> is a generalization of <code>reduce</code> to key-value pairs. For each group of consecutive keys in the range <code>[keys&#95;first, keys&#95;last)</code> that are equal, <code>reduce&#95;by&#95;key</code> copies the first element of the group to the <code>keys&#95;output</code>. The corresponding values in the range are reduced using the <code>plus</code> and the result copied to <code>values&#95;output</code>.

This version of <code>reduce&#95;by&#95;key</code> uses the function object <code>binary&#95;pred</code> to test for equality and <code>plus</code> to reduce values with equal keys.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>reduce&#95;by&#95;key</code> to compact a sequence of key/value pairs and sum values with equal keys using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
int C[N];                         // output keys
int D[N];                         // output values

thrust::pair<int*,int*> new_end;
thrust::equal_to<int> binary_pred;
new_end = thrust::reduce_by_key(thrust::host, A, A + N, B, C, D, binary_pred);

// The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
// The first four values in D are now {9, 21, 9, 3} and new_end.second - D is 4.
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
* **`keys_output`** The beginning of the output key range. 
* **`values_output`** The beginning of the output value range. 
* **`binary_pred`** The binary predicate used to determine equality. 

**Preconditions**:
The input ranges shall not overlap either output range.

**Returns**:
A pair of iterators at end of the ranges <code>[keys&#95;output, keys&#95;output&#95;last)</code> and <code>[values&#95;output, values&#95;output&#95;last)</code>.

**See**:
* reduce 
* unique_copy 
* unique_by_key 
* unique_by_key_copy 

<h3 id="function-reduce-by-key">
Function <code>thrust::reduce&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>reduce_by_key</b>(InputIterator1 keys_first,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last,</span>
<span>&nbsp;&nbsp;InputIterator2 values_first,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_output,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_output,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred);</span></code>
<code>reduce&#95;by&#95;key</code> is a generalization of <code>reduce</code> to key-value pairs. For each group of consecutive keys in the range <code>[keys&#95;first, keys&#95;last)</code> that are equal, <code>reduce&#95;by&#95;key</code> copies the first element of the group to the <code>keys&#95;output</code>. The corresponding values in the range are reduced using the <code>plus</code> and the result copied to <code>values&#95;output</code>.

This version of <code>reduce&#95;by&#95;key</code> uses the function object <code>binary&#95;pred</code> to test for equality and <code>plus</code> to reduce values with equal keys.


The following code snippet demonstrates how to use <code>reduce&#95;by&#95;key</code> to compact a sequence of key/value pairs and sum values with equal keys.



```cpp
#include <thrust/reduce.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
int C[N];                         // output keys
int D[N];                         // output values

thrust::pair<int*,int*> new_end;
thrust::equal_to<int> binary_pred;
new_end = thrust::reduce_by_key(A, A + N, B, C, D, binary_pred);

// The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
// The first four values in D are now {9, 21, 9, 3} and new_end.second - D is 4.
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
* **`keys_output`** The beginning of the output key range. 
* **`values_output`** The beginning of the output value range. 
* **`binary_pred`** The binary predicate used to determine equality. 

**Preconditions**:
The input ranges shall not overlap either output range.

**Returns**:
A pair of iterators at end of the ranges <code>[keys&#95;output, keys&#95;output&#95;last)</code> and <code>[values&#95;output, values&#95;output&#95;last)</code>.

**See**:
* reduce 
* unique_copy 
* unique_by_key 
* unique_by_key_copy 

<h3 id="function-reduce-by-key">
Function <code>thrust::reduce&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate,</span>
<span>&nbsp;&nbsp;typename BinaryFunction&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>reduce_by_key</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_first,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last,</span>
<span>&nbsp;&nbsp;InputIterator2 values_first,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_output,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_output,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred,</span>
<span>&nbsp;&nbsp;BinaryFunction binary_op);</span></code>
<code>reduce&#95;by&#95;key</code> is a generalization of <code>reduce</code> to key-value pairs. For each group of consecutive keys in the range <code>[keys&#95;first, keys&#95;last)</code> that are equal, <code>reduce&#95;by&#95;key</code> copies the first element of the group to the <code>keys&#95;output</code>. The corresponding values in the range are reduced using the <code>BinaryFunction</code><code>binary&#95;op</code> and the result copied to <code>values&#95;output</code>. Specifically, if consecutive key iterators <code>i</code> and <code></code>(i + 1) are such that <code>binary&#95;pred(&#42;i, &#42;(i+1))</code> is <code>true</code>, then the corresponding values are reduced to a single value with <code>binary&#95;op</code>.

This version of <code>reduce&#95;by&#95;key</code> uses the function object <code>binary&#95;pred</code> to test for equality and <code>binary&#95;op</code> to reduce values with equal keys.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>reduce&#95;by&#95;key</code> to compact a sequence of key/value pairs and sum values with equal keys using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
int C[N];                         // output keys
int D[N];                         // output values

thrust::pair<int*,int*> new_end;
thrust::equal_to<int> binary_pred;
thrust::plus<int> binary_op;
new_end = thrust::reduce_by_key(thrust::host, A, A + N, B, C, D, binary_pred, binary_op);

// The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
// The first four values in D are now {9, 21, 9, 3} and new_end.second - D is 4.
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and and <code>InputIterator1's</code><code>value&#95;type</code> is convertible to <code>OutputIterator1's</code><code>value&#95;type</code>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>OutputIterator2's</code><code>value&#95;type</code>. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>. 
* **`BinaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a> and <code>BinaryFunction's</code><code>result&#95;type</code> is convertible to <code>OutputIterator2's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`keys_first`** The beginning of the input key range. 
* **`keys_last`** The end of the input key range. 
* **`values_first`** The beginning of the input value range. 
* **`keys_output`** The beginning of the output key range. 
* **`values_output`** The beginning of the output value range. 
* **`binary_pred`** The binary predicate used to determine equality. 
* **`binary_op`** The binary function used to accumulate values. 

**Preconditions**:
The input ranges shall not overlap either output range.

**Returns**:
A pair of iterators at end of the ranges <code>[keys&#95;output, keys&#95;output&#95;last)</code> and <code>[values&#95;output, values&#95;output&#95;last)</code>.

**See**:
* reduce 
* unique_copy 
* unique_by_key 
* unique_by_key_copy 

<h3 id="function-reduce-by-key">
Function <code>thrust::reduce&#95;by&#95;key</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename BinaryPredicate,</span>
<span>&nbsp;&nbsp;typename BinaryFunction&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>reduce_by_key</b>(InputIterator1 keys_first,</span>
<span>&nbsp;&nbsp;InputIterator1 keys_last,</span>
<span>&nbsp;&nbsp;InputIterator2 values_first,</span>
<span>&nbsp;&nbsp;OutputIterator1 keys_output,</span>
<span>&nbsp;&nbsp;OutputIterator2 values_output,</span>
<span>&nbsp;&nbsp;BinaryPredicate binary_pred,</span>
<span>&nbsp;&nbsp;BinaryFunction binary_op);</span></code>
<code>reduce&#95;by&#95;key</code> is a generalization of <code>reduce</code> to key-value pairs. For each group of consecutive keys in the range <code>[keys&#95;first, keys&#95;last)</code> that are equal, <code>reduce&#95;by&#95;key</code> copies the first element of the group to the <code>keys&#95;output</code>. The corresponding values in the range are reduced using the <code>BinaryFunction</code><code>binary&#95;op</code> and the result copied to <code>values&#95;output</code>. Specifically, if consecutive key iterators <code>i</code> and <code></code>(i + 1) are such that <code>binary&#95;pred(&#42;i, &#42;(i+1))</code> is <code>true</code>, then the corresponding values are reduced to a single value with <code>binary&#95;op</code>.

This version of <code>reduce&#95;by&#95;key</code> uses the function object <code>binary&#95;pred</code> to test for equality and <code>binary&#95;op</code> to reduce values with equal keys.


The following code snippet demonstrates how to use <code>reduce&#95;by&#95;key</code> to compact a sequence of key/value pairs and sum values with equal keys.



```cpp
#include <thrust/reduce.h>
...
const int N = 7;
int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
int C[N];                         // output keys
int D[N];                         // output values

thrust::pair<int*,int*> new_end;
thrust::equal_to<int> binary_pred;
thrust::plus<int> binary_op;
new_end = thrust::reduce_by_key(A, A + N, B, C, D, binary_pred, binary_op);

// The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
// The first four values in D are now {9, 21, 9, 3} and new_end.second - D is 4.
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and and <code>InputIterator1's</code><code>value&#95;type</code> is convertible to <code>OutputIterator1's</code><code>value&#95;type</code>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a> and and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>OutputIterator2's</code><code>value&#95;type</code>. 
* **`BinaryPredicate`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary Predicate</a>. 
* **`BinaryFunction`** is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a> and <code>BinaryFunction's</code><code>result&#95;type</code> is convertible to <code>OutputIterator2's</code><code>value&#95;type</code>.

**Function Parameters**:
* **`keys_first`** The beginning of the input key range. 
* **`keys_last`** The end of the input key range. 
* **`values_first`** The beginning of the input value range. 
* **`keys_output`** The beginning of the output key range. 
* **`values_output`** The beginning of the output value range. 
* **`binary_pred`** The binary predicate used to determine equality. 
* **`binary_op`** The binary function used to accumulate values. 

**Preconditions**:
The input ranges shall not overlap either output range.

**Returns**:
A pair of iterators at end of the ranges <code>[keys&#95;output, keys&#95;output&#95;last)</code> and <code>[values&#95;output, values&#95;output&#95;last)</code>.

**See**:
* reduce 
* unique_copy 
* unique_by_key 
* unique_by_key_copy 


