---
title: Partitioning
parent: Reordering
grand_parent: Algorithms
nav_exclude: false
has_children: true
has_toc: false
---

# Partitioning

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__partitioning.html#function-partition">thrust::partition</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__partitioning.html#function-partition">thrust::partition</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__partitioning.html#function-partition">thrust::partition</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator stencil,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__partitioning.html#function-partition">thrust::partition</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator stencil,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__partitioning.html#function-partition-copy">thrust::partition&#95;copy</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator1 out_true,</span>
<span>&nbsp;&nbsp;OutputIterator2 out_false,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__partitioning.html#function-partition-copy">thrust::partition&#95;copy</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator1 out_true,</span>
<span>&nbsp;&nbsp;OutputIterator2 out_false,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__partitioning.html#function-partition-copy">thrust::partition&#95;copy</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;OutputIterator1 out_true,</span>
<span>&nbsp;&nbsp;OutputIterator2 out_false,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__partitioning.html#function-partition-copy">thrust::partition&#95;copy</a></b>(InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;OutputIterator1 out_true,</span>
<span>&nbsp;&nbsp;OutputIterator2 out_false,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__partitioning.html#function-stable-partition">thrust::stable&#95;partition</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__partitioning.html#function-stable-partition">thrust::stable&#95;partition</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__partitioning.html#function-stable-partition">thrust::stable&#95;partition</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator stencil,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>ForwardIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__partitioning.html#function-stable-partition">thrust::stable&#95;partition</a></b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator stencil,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__partitioning.html#function-stable-partition-copy">thrust::stable&#95;partition&#95;copy</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator1 out_true,</span>
<span>&nbsp;&nbsp;OutputIterator2 out_false,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__partitioning.html#function-stable-partition-copy">thrust::stable&#95;partition&#95;copy</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator1 out_true,</span>
<span>&nbsp;&nbsp;OutputIterator2 out_false,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__partitioning.html#function-stable-partition-copy">thrust::stable&#95;partition&#95;copy</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;OutputIterator1 out_true,</span>
<span>&nbsp;&nbsp;OutputIterator2 out_false,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__partitioning.html#function-stable-partition-copy">thrust::stable&#95;partition&#95;copy</a></b>(InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;OutputIterator1 out_true,</span>
<span>&nbsp;&nbsp;OutputIterator2 out_false,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
</code>

## Functions

<h3 id="function-partition">
Function <code>thrust::partition</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>partition</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>partition</code> reorders the elements <code>[first, last)</code> based on the function object <code>pred</code>, such that all of the elements that satisfy <code>pred</code> precede the elements that fail to satisfy it. The postcondition is that, for some iterator <code>middle</code> in the range <code>[first, last)</code>, <code>pred(&#42;i)</code> is <code>true</code> for every iterator <code>i</code> in the range <code>[first,middle)</code> and <code>false</code> for every iterator <code>i</code> in the range <code>[middle, last)</code>. The return value of <code>partition</code> is <code>middle</code>.

Note that the relative order of elements in the two reordered sequences is not necessarily the same as it was in the original sequence. A different algorithm, <code>stable&#95;partition</code>, does guarantee to preserve the relative order.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>partition</code> to reorder a sequence so that even numbers precede odd numbers using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/partition.h>
#include <thrust/execution_policy.h>
...
struct is_even
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return (x % 2) == 0;
  }
};
...
int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
const int N = sizeof(A)/sizeof(int);
thrust::partition(thrust::host,
                  A, A + N,
                  is_even());
// A is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>, and <code>ForwardIterator</code> is mutable. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence to reorder. 
* **`last`** The end of the sequence to reorder. 
* **`pred`** A function object which decides to which partition each element of the sequence <code>[first, last)</code> belongs. 

**Returns**:
An iterator referring to the first element of the second partition, that is, the sequence of the elements which do not satisfy <code>pred</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/partition">https://en.cppreference.com/w/cpp/algorithm/partition</a>
* <code>stable&#95;partition</code>
* <code>partition&#95;copy</code>

<h3 id="function-partition">
Function <code>thrust::partition</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>ForwardIterator </span><span><b>partition</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>partition</code> reorders the elements <code>[first, last)</code> based on the function object <code>pred</code>, such that all of the elements that satisfy <code>pred</code> precede the elements that fail to satisfy it. The postcondition is that, for some iterator <code>middle</code> in the range <code>[first, last)</code>, <code>pred(&#42;i)</code> is <code>true</code> for every iterator <code>i</code> in the range <code>[first,middle)</code> and <code>false</code> for every iterator <code>i</code> in the range <code>[middle, last)</code>. The return value of <code>partition</code> is <code>middle</code>.

Note that the relative order of elements in the two reordered sequences is not necessarily the same as it was in the original sequence. A different algorithm, <code>stable&#95;partition</code>, does guarantee to preserve the relative order.


The following code snippet demonstrates how to use <code>partition</code> to reorder a sequence so that even numbers precede odd numbers.



```cpp
#include <thrust/partition.h>
...
struct is_even
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return (x % 2) == 0;
  }
};
...
int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
const int N = sizeof(A)/sizeof(int);
thrust::partition(A, A + N,
                   is_even());
// A is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>, and <code>ForwardIterator</code> is mutable. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first`** The beginning of the sequence to reorder. 
* **`last`** The end of the sequence to reorder. 
* **`pred`** A function object which decides to which partition each element of the sequence <code>[first, last)</code> belongs. 

**Returns**:
An iterator referring to the first element of the second partition, that is, the sequence of the elements which do not satisfy <code>pred</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/partition">https://en.cppreference.com/w/cpp/algorithm/partition</a>
* <code>stable&#95;partition</code>
* <code>partition&#95;copy</code>

<h3 id="function-partition">
Function <code>thrust::partition</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>partition</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator stencil,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>partition</code> reorders the elements <code>[first, last)</code> based on the function object <code>pred</code> applied to a stencil range <code>[stencil, stencil + (last - first))</code>, such that all of the elements whose corresponding stencil element satisfies <code>pred</code> precede all of the elements whose corresponding stencil element fails to satisfy it. The postcondition is that, for some iterator <code>middle</code> in the range <code>[first, last)</code>, <code>pred(&#42;stencil&#95;i)</code> is <code>true</code> for every iterator <code>stencil&#95;i</code> in the range <code>[stencil,stencil + (middle - first))</code> and <code>false</code> for every iterator <code>stencil&#95;i</code> in the range <code>[stencil + (middle - first), stencil + (last - first))</code>. The return value of <code>stable&#95;partition</code> is <code>middle</code>.

Note that the relative order of elements in the two reordered sequences is not necessarily the same as it was in the original sequence. A different algorithm, <code>stable&#95;partition</code>, does guarantee to preserve the relative order.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>partition</code> to reorder a sequence so that even numbers precede odd numbers using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/partition.h>
#include <thrust/execution_policy.h>
...
struct is_even
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return (x % 2) == 0;
  }
};
...
int A[] = {0, 1, 0, 1, 0, 1, 0, 1, 0,  1};
int S[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
const int N = sizeof(A)/sizeof(int);
thrust::partition(thrust::host, A, A + N, S, is_even());
// A is now {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}
// S is unmodified
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence to reorder. 
* **`last`** The end of the sequence to reorder. 
* **`stencil`** The beginning of the stencil sequence. 
* **`pred`** A function object which decides to which partition each element of the sequence <code>[first, last)</code> belongs. 

**Preconditions**:
The ranges <code>[first,last)</code> and <code>[stencil, stencil + (last - first))</code> shall not overlap.

**Returns**:
An iterator referring to the first element of the second partition, that is, the sequence of the elements whose stencil elements do not satisfy <code>pred</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/partition">https://en.cppreference.com/w/cpp/algorithm/partition</a>
* <code>stable&#95;partition</code>
* <code>partition&#95;copy</code>

<h3 id="function-partition">
Function <code>thrust::partition</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>ForwardIterator </span><span><b>partition</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator stencil,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>partition</code> reorders the elements <code>[first, last)</code> based on the function object <code>pred</code> applied to a stencil range <code>[stencil, stencil + (last - first))</code>, such that all of the elements whose corresponding stencil element satisfies <code>pred</code> precede all of the elements whose corresponding stencil element fails to satisfy it. The postcondition is that, for some iterator <code>middle</code> in the range <code>[first, last)</code>, <code>pred(&#42;stencil&#95;i)</code> is <code>true</code> for every iterator <code>stencil&#95;i</code> in the range <code>[stencil,stencil + (middle - first))</code> and <code>false</code> for every iterator <code>stencil&#95;i</code> in the range <code>[stencil + (middle - first), stencil + (last - first))</code>. The return value of <code>stable&#95;partition</code> is <code>middle</code>.

Note that the relative order of elements in the two reordered sequences is not necessarily the same as it was in the original sequence. A different algorithm, <code>stable&#95;partition</code>, does guarantee to preserve the relative order.


The following code snippet demonstrates how to use <code>partition</code> to reorder a sequence so that even numbers precede odd numbers.



```cpp
#include <thrust/partition.h>
...
struct is_even
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return (x % 2) == 0;
  }
};
...
int A[] = {0, 1, 0, 1, 0, 1, 0, 1, 0,  1};
int S[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
const int N = sizeof(A)/sizeof(int);
thrust::partition(A, A + N, S, is_even());
// A is now {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}
// S is unmodified
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first`** The beginning of the sequence to reorder. 
* **`last`** The end of the sequence to reorder. 
* **`stencil`** The beginning of the stencil sequence. 
* **`pred`** A function object which decides to which partition each element of the sequence <code>[first, last)</code> belongs. 

**Preconditions**:
The ranges <code>[first,last)</code> and <code>[stencil, stencil + (last - first))</code> shall not overlap.

**Returns**:
An iterator referring to the first element of the second partition, that is, the sequence of the elements whose stencil elements do not satisfy <code>pred</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/partition">https://en.cppreference.com/w/cpp/algorithm/partition</a>
* <code>stable&#95;partition</code>
* <code>partition&#95;copy</code>

<h3 id="function-partition-copy">
Function <code>thrust::partition&#95;copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>partition_copy</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator1 out_true,</span>
<span>&nbsp;&nbsp;OutputIterator2 out_false,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>partition&#95;copy</code> differs from <code>partition</code> only in that the reordered sequence is written to difference output sequences, rather than in place.

<code>partition&#95;copy</code> copies the elements <code>[first, last)</code> based on the function object <code>pred</code>. All of the elements that satisfy <code>pred</code> are copied to the range beginning at <code>out&#95;true</code> and all the elements that fail to satisfy it are copied to the range beginning at <code>out&#95;false</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>partition&#95;copy</code> to separate a sequence into two output sequences of even and odd numbers using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/partition.h>
#include <thrust/execution_policy.h>
...
struct is_even
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return (x % 2) == 0;
  }
};
...
int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
int result[10];
const int N = sizeof(A)/sizeof(int);
int *evens = result;
int *odds  = result + 5;
thrust::partition_copy(thrust::host, A, A + N, evens, odds, is_even());
// A remains {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
// result is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
// evens points to {2, 4, 6, 8, 10}
// odds points to {1, 3, 5, 7, 9}
```

**Note**:
The relative order of elements in the two reordered sequences is not necessarily the same as it was in the original sequence. A different algorithm, <code>stable&#95;partition&#95;copy</code>, does guarantee to preserve the relative order.

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code> and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>OutputIterator1</code> and <code>OutputIterator2's</code><code>value&#95;types</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence to reorder. 
* **`last`** The end of the sequence to reorder. 
* **`out_true`** The destination of the resulting sequence of elements which satisfy <code>pred</code>. 
* **`out_false`** The destination of the resulting sequence of elements which fail to satisfy <code>pred</code>. 
* **`pred`** A function object which decides to which partition each element of the sequence <code>[first, last)</code> belongs. 

**Preconditions**:
The input range shall not overlap with either output range.

**Returns**:
A <code>pair</code> p such that <code>p.first</code> is the end of the output range beginning at <code>out&#95;true</code> and <code>p.second</code> is the end of the output range beginning at <code>out&#95;false</code>.

**See**:
* <a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf">http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf</a>
* <code>stable&#95;partition&#95;copy</code>
* <code>partition</code>

<h3 id="function-partition-copy">
Function <code>thrust::partition&#95;copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>partition_copy</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator1 out_true,</span>
<span>&nbsp;&nbsp;OutputIterator2 out_false,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>partition&#95;copy</code> differs from <code>partition</code> only in that the reordered sequence is written to difference output sequences, rather than in place.

<code>partition&#95;copy</code> copies the elements <code>[first, last)</code> based on the function object <code>pred</code>. All of the elements that satisfy <code>pred</code> are copied to the range beginning at <code>out&#95;true</code> and all the elements that fail to satisfy it are copied to the range beginning at <code>out&#95;false</code>.


The following code snippet demonstrates how to use <code>partition&#95;copy</code> to separate a sequence into two output sequences of even and odd numbers.



```cpp
#include <thrust/partition.h>
...
struct is_even
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return (x % 2) == 0;
  }
};
...
int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
int result[10];
const int N = sizeof(A)/sizeof(int);
int *evens = result;
int *odds  = result + 5;
thrust::partition_copy(A, A + N, evens, odds, is_even());
// A remains {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
// result is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
// evens points to {2, 4, 6, 8, 10}
// odds points to {1, 3, 5, 7, 9}
```

**Note**:
The relative order of elements in the two reordered sequences is not necessarily the same as it was in the original sequence. A different algorithm, <code>stable&#95;partition&#95;copy</code>, does guarantee to preserve the relative order.

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code> and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>OutputIterator1</code> and <code>OutputIterator2's</code><code>value&#95;types</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first`** The beginning of the sequence to reorder. 
* **`last`** The end of the sequence to reorder. 
* **`out_true`** The destination of the resulting sequence of elements which satisfy <code>pred</code>. 
* **`out_false`** The destination of the resulting sequence of elements which fail to satisfy <code>pred</code>. 
* **`pred`** A function object which decides to which partition each element of the sequence <code>[first, last)</code> belongs. 

**Preconditions**:
The input range shall not overlap with either output range.

**Returns**:
A <code>pair</code> p such that <code>p.first</code> is the end of the output range beginning at <code>out&#95;true</code> and <code>p.second</code> is the end of the output range beginning at <code>out&#95;false</code>.

**See**:
* <a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf">http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf</a>
* <code>stable&#95;partition&#95;copy</code>
* <code>partition</code>

<h3 id="function-partition-copy">
Function <code>thrust::partition&#95;copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>partition_copy</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;OutputIterator1 out_true,</span>
<span>&nbsp;&nbsp;OutputIterator2 out_false,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>partition&#95;copy</code> differs from <code>partition</code> only in that the reordered sequence is written to difference output sequences, rather than in place.

<code>partition&#95;copy</code> copies the elements <code>[first, last)</code> based on the function object <code>pred</code> which is applied to a range of stencil elements. All of the elements whose corresponding stencil element satisfies <code>pred</code> are copied to the range beginning at <code>out&#95;true</code> and all the elements whose stencil element fails to satisfy it are copied to the range beginning at <code>out&#95;false</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>partition&#95;copy</code> to separate a sequence into two output sequences of even and odd numbers using the <code>thrust::host</code> execution policy for parallelization.



```cpp
#include <thrust/partition.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...
int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
int S[] = {0, 1, 0, 1, 0, 1, 0, 1, 0,  1};
int result[10];
const int N = sizeof(A)/sizeof(int);
int *evens = result;
int *odds  = result + 5;
thrust::stable_partition_copy(thrust::host, A, A + N, S, evens, odds, thrust::identity<int>());
// A remains {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
// S remains {0, 1, 0, 1, 0, 1, 0, 1, 0,  1}
// result is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
// evens points to {2, 4, 6, 8, 10}
// odds points to {1, 3, 5, 7, 9}
```

**Note**:
The relative order of elements in the two reordered sequences is not necessarily the same as it was in the original sequence. A different algorithm, <code>stable&#95;partition&#95;copy</code>, does guarantee to preserve the relative order.

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>OutputIterator1</code> and <code>OutputIterator2's</code><code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence to reorder. 
* **`last`** The end of the sequence to reorder. 
* **`stencil`** The beginning of the stencil sequence. 
* **`out_true`** The destination of the resulting sequence of elements which satisfy <code>pred</code>. 
* **`out_false`** The destination of the resulting sequence of elements which fail to satisfy <code>pred</code>. 
* **`pred`** A function object which decides to which partition each element of the sequence <code>[first, last)</code> belongs. 

**Preconditions**:
The input ranges shall not overlap with either output range.

**Returns**:
A <code>pair</code> p such that <code>p.first</code> is the end of the output range beginning at <code>out&#95;true</code> and <code>p.second</code> is the end of the output range beginning at <code>out&#95;false</code>.

**See**:
* <a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf">http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf</a>
* <code>stable&#95;partition&#95;copy</code>
* <code>partition</code>

<h3 id="function-partition-copy">
Function <code>thrust::partition&#95;copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>partition_copy</b>(InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;OutputIterator1 out_true,</span>
<span>&nbsp;&nbsp;OutputIterator2 out_false,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>partition&#95;copy</code> differs from <code>partition</code> only in that the reordered sequence is written to difference output sequences, rather than in place.

<code>partition&#95;copy</code> copies the elements <code>[first, last)</code> based on the function object <code>pred</code> which is applied to a range of stencil elements. All of the elements whose corresponding stencil element satisfies <code>pred</code> are copied to the range beginning at <code>out&#95;true</code> and all the elements whose stencil element fails to satisfy it are copied to the range beginning at <code>out&#95;false</code>.


The following code snippet demonstrates how to use <code>partition&#95;copy</code> to separate a sequence into two output sequences of even and odd numbers.



```cpp
#include <thrust/partition.h>
#include <thrust/functional.h>
...
int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
int S[] = {0, 1, 0, 1, 0, 1, 0, 1, 0,  1};
int result[10];
const int N = sizeof(A)/sizeof(int);
int *evens = result;
int *odds  = result + 5;
thrust::stable_partition_copy(A, A + N, S, evens, odds, thrust::identity<int>());
// A remains {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
// S remains {0, 1, 0, 1, 0, 1, 0, 1, 0,  1}
// result is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
// evens points to {2, 4, 6, 8, 10}
// odds points to {1, 3, 5, 7, 9}
```

**Note**:
The relative order of elements in the two reordered sequences is not necessarily the same as it was in the original sequence. A different algorithm, <code>stable&#95;partition&#95;copy</code>, does guarantee to preserve the relative order.

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>OutputIterator1</code> and <code>OutputIterator2's</code><code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first`** The beginning of the sequence to reorder. 
* **`last`** The end of the sequence to reorder. 
* **`stencil`** The beginning of the stencil sequence. 
* **`out_true`** The destination of the resulting sequence of elements which satisfy <code>pred</code>. 
* **`out_false`** The destination of the resulting sequence of elements which fail to satisfy <code>pred</code>. 
* **`pred`** A function object which decides to which partition each element of the sequence <code>[first, last)</code> belongs. 

**Preconditions**:
The input ranges shall not overlap with either output range.

**Returns**:
A <code>pair</code> p such that <code>p.first</code> is the end of the output range beginning at <code>out&#95;true</code> and <code>p.second</code> is the end of the output range beginning at <code>out&#95;false</code>.

**See**:
* <a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf">http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf</a>
* <code>stable&#95;partition&#95;copy</code>
* <code>partition</code>

<h3 id="function-stable-partition">
Function <code>thrust::stable&#95;partition</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>stable_partition</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>stable&#95;partition</code> is much like <code>partition</code> : it reorders the elements in the range <code>[first, last)</code> based on the function object <code>pred</code>, such that all of the elements that satisfy <code>pred</code> precede all of the elements that fail to satisfy it. The postcondition is that, for some iterator <code>middle</code> in the range <code>[first, last)</code>, <code>pred(&#42;i)</code> is <code>true</code> for every iterator <code>i</code> in the range <code>[first,middle)</code> and <code>false</code> for every iterator <code>i</code> in the range <code>[middle, last)</code>. The return value of <code>stable&#95;partition</code> is <code>middle</code>.

<code>stable&#95;partition</code> differs from <code>partition</code> in that <code>stable&#95;partition</code> is guaranteed to preserve relative order. That is, if <code>x</code> and <code>y</code> are elements in <code>[first, last)</code>, and <code>stencil&#95;x</code> and <code>stencil&#95;y</code> are the stencil elements in corresponding positions within <code>[stencil, stencil + (last - first))</code>, and <code>pred(stencil&#95;x) == pred(stencil&#95;y)</code>, and if <code>x</code> precedes <code>y</code>, then it will still be true after <code>stable&#95;partition</code> that <code>x</code> precedes <code>y</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>stable&#95;partition</code> to reorder a sequence so that even numbers precede odd numbers using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/partition.h>
#include <thrust/execution_policy.h>
...
struct is_even
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return (x % 2) == 0;
  }
};
...
int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
const int N = sizeof(A)/sizeof(int);
thrust::stable_partition(thrust::host,
                         A, A + N,
                         is_even());
// A is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>, and <code>ForwardIterator</code> is mutable. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The first element of the sequence to reorder. 
* **`last`** One position past the last element of the sequence to reorder. 
* **`pred`** A function object which decides to which partition each element of the sequence <code>[first, last)</code> belongs. 

**Returns**:
An iterator referring to the first element of the second partition, that is, the sequence of the elements which do not satisfy pred.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/stable_partition">https://en.cppreference.com/w/cpp/algorithm/stable_partition</a>
* <code>partition</code>
* <code>stable&#95;partition&#95;copy</code>

<h3 id="function-stable-partition">
Function <code>thrust::stable&#95;partition</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>ForwardIterator </span><span><b>stable_partition</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>stable&#95;partition</code> is much like <code>partition</code> : it reorders the elements in the range <code>[first, last)</code> based on the function object <code>pred</code>, such that all of the elements that satisfy <code>pred</code> precede all of the elements that fail to satisfy it. The postcondition is that, for some iterator <code>middle</code> in the range <code>[first, last)</code>, <code>pred(&#42;i)</code> is <code>true</code> for every iterator <code>i</code> in the range <code>[first,middle)</code> and <code>false</code> for every iterator <code>i</code> in the range <code>[middle, last)</code>. The return value of <code>stable&#95;partition</code> is <code>middle</code>.

<code>stable&#95;partition</code> differs from <code>partition</code> in that <code>stable&#95;partition</code> is guaranteed to preserve relative order. That is, if <code>x</code> and <code>y</code> are elements in <code>[first, last)</code>, and <code>stencil&#95;x</code> and <code>stencil&#95;y</code> are the stencil elements in corresponding positions within <code>[stencil, stencil + (last - first))</code>, and <code>pred(stencil&#95;x) == pred(stencil&#95;y)</code>, and if <code>x</code> precedes <code>y</code>, then it will still be true after <code>stable&#95;partition</code> that <code>x</code> precedes <code>y</code>.


The following code snippet demonstrates how to use <code>stable&#95;partition</code> to reorder a sequence so that even numbers precede odd numbers.



```cpp
#include <thrust/partition.h>
...
struct is_even
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return (x % 2) == 0;
  }
};
...
int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
const int N = sizeof(A)/sizeof(int);
thrust::stable_partition(A, A + N,
                          is_even());
// A is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>, and <code>ForwardIterator</code> is mutable. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first`** The first element of the sequence to reorder. 
* **`last`** One position past the last element of the sequence to reorder. 
* **`pred`** A function object which decides to which partition each element of the sequence <code>[first, last)</code> belongs. 

**Returns**:
An iterator referring to the first element of the second partition, that is, the sequence of the elements which do not satisfy pred.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/stable_partition">https://en.cppreference.com/w/cpp/algorithm/stable_partition</a>
* <code>partition</code>
* <code>stable&#95;partition&#95;copy</code>

<h3 id="function-stable-partition">
Function <code>thrust::stable&#95;partition</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ ForwardIterator </span><span><b>stable_partition</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator stencil,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>stable&#95;partition</code> is much like <code>partition:</code> it reorders the elements in the range <code>[first, last)</code> based on the function object <code>pred</code> applied to a stencil range <code>[stencil, stencil + (last - first))</code>, such that all of the elements whose corresponding stencil element satisfies <code>pred</code> precede all of the elements whose corresponding stencil element fails to satisfy it. The postcondition is that, for some iterator <code>middle</code> in the range <code>[first, last)</code>, <code>pred(&#42;stencil&#95;i)</code> is <code>true</code> for every iterator <code>stencil&#95;i</code> in the range <code>[stencil,stencil + (middle - first))</code> and <code>false</code> for every iterator <code>stencil&#95;i</code> in the range <code>[stencil + (middle - first), stencil + (last - first))</code>. The return value of <code>stable&#95;partition</code> is <code>middle</code>.

<code>stable&#95;partition</code> differs from <code>partition</code> in that <code>stable&#95;partition</code> is guaranteed to preserve relative order. That is, if <code>x</code> and <code>y</code> are elements in <code>[first, last)</code>, such that <code>pred(x) == pred(y)</code>, and if <code>x</code> precedes <code>y</code>, then it will still be true after <code>stable&#95;partition</code> that <code>x</code> precedes <code>y</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>stable&#95;partition</code> to reorder a sequence so that even numbers precede odd numbers using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/partition.h>
#include <thrust/execution_policy.h>
...
struct is_even
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return (x % 2) == 0;
  }
};
...
int A[] = {0, 1, 0, 1, 0, 1, 0, 1, 0,  1};
int S[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
const int N = sizeof(A)/sizeof(int);
thrust::stable_partition(thrust::host, A, A + N, S, is_even());
// A is now {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}
// S is unmodified
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The first element of the sequence to reorder. 
* **`last`** One position past the last element of the sequence to reorder. 
* **`stencil`** The beginning of the stencil sequence. 
* **`pred`** A function object which decides to which partition each element of the sequence <code>[first, last)</code> belongs. 

**Preconditions**:
The range <code>[first, last)</code> shall not overlap with the range <code>[stencil, stencil + (last - first))</code>.

**Returns**:
An iterator referring to the first element of the second partition, that is, the sequence of the elements whose stencil elements do not satisfy <code>pred</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/stable_partition">https://en.cppreference.com/w/cpp/algorithm/stable_partition</a>
* <code>partition</code>
* <code>stable&#95;partition&#95;copy</code>

<h3 id="function-stable-partition">
Function <code>thrust::stable&#95;partition</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ForwardIterator,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>ForwardIterator </span><span><b>stable_partition</b>(ForwardIterator first,</span>
<span>&nbsp;&nbsp;ForwardIterator last,</span>
<span>&nbsp;&nbsp;InputIterator stencil,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>stable&#95;partition</code> is much like <code>partition:</code> it reorders the elements in the range <code>[first, last)</code> based on the function object <code>pred</code> applied to a stencil range <code>[stencil, stencil + (last - first))</code>, such that all of the elements whose corresponding stencil element satisfies <code>pred</code> precede all of the elements whose corresponding stencil element fails to satisfy it. The postcondition is that, for some iterator <code>middle</code> in the range <code>[first, last)</code>, <code>pred(&#42;stencil&#95;i)</code> is <code>true</code> for every iterator <code>stencil&#95;i</code> in the range <code>[stencil,stencil + (middle - first))</code> and <code>false</code> for every iterator <code>stencil&#95;i</code> in the range <code>[stencil + (middle - first), stencil + (last - first))</code>. The return value of <code>stable&#95;partition</code> is <code>middle</code>.

<code>stable&#95;partition</code> differs from <code>partition</code> in that <code>stable&#95;partition</code> is guaranteed to preserve relative order. That is, if <code>x</code> and <code>y</code> are elements in <code>[first, last)</code>, such that <code>pred(x) == pred(y)</code>, and if <code>x</code> precedes <code>y</code>, then it will still be true after <code>stable&#95;partition</code> that <code>x</code> precedes <code>y</code>.


The following code snippet demonstrates how to use <code>stable&#95;partition</code> to reorder a sequence so that even numbers precede odd numbers.



```cpp
#include <thrust/partition.h>
...
struct is_even
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return (x % 2) == 0;
  }
};
...
int A[] = {0, 1, 0, 1, 0, 1, 0, 1, 0,  1};
int S[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
const int N = sizeof(A)/sizeof(int);
thrust::stable_partition(A, A + N, S, is_even());
// A is now {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}
// S is unmodified
```

**Template Parameters**:
* **`ForwardIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>, and <code>ForwardIterator</code> is mutable. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first`** The first element of the sequence to reorder. 
* **`last`** One position past the last element of the sequence to reorder. 
* **`stencil`** The beginning of the stencil sequence. 
* **`pred`** A function object which decides to which partition each element of the sequence <code>[first, last)</code> belongs. 

**Preconditions**:
The range <code>[first, last)</code> shall not overlap with the range <code>[stencil, stencil + (last - first))</code>.

**Returns**:
An iterator referring to the first element of the second partition, that is, the sequence of the elements whose stencil elements do not satisfy <code>pred</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/algorithm/stable_partition">https://en.cppreference.com/w/cpp/algorithm/stable_partition</a>
* <code>partition</code>
* <code>stable&#95;partition&#95;copy</code>

<h3 id="function-stable-partition-copy">
Function <code>thrust::stable&#95;partition&#95;copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>stable_partition_copy</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator1 out_true,</span>
<span>&nbsp;&nbsp;OutputIterator2 out_false,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>stable&#95;partition&#95;copy</code> differs from <code>stable&#95;partition</code> only in that the reordered sequence is written to different output sequences, rather than in place.

<code>stable&#95;partition&#95;copy</code> copies the elements <code>[first, last)</code> based on the function object <code>pred</code>. All of the elements that satisfy <code>pred</code> are copied to the range beginning at <code>out&#95;true</code> and all the elements that fail to satisfy it are copied to the range beginning at <code>out&#95;false</code>.

<code>stable&#95;partition&#95;copy</code> differs from <code>partition&#95;copy</code> in that <code>stable&#95;partition&#95;copy</code> is guaranteed to preserve relative order. That is, if <code>x</code> and <code>y</code> are elements in <code>[first, last)</code>, such that <code>pred(x) == pred(y)</code>, and if <code>x</code> precedes <code>y</code>, then it will still be true after <code>stable&#95;partition&#95;copy</code> that <code>x</code> precedes <code>y</code> in the output.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>stable&#95;partition&#95;copy</code> to reorder a sequence so that even numbers precede odd numbers using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/partition.h>
#include <thrust/execution_policy.h>
...
struct is_even
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return (x % 2) == 0;
  }
};
...
int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
int result[10];
const int N = sizeof(A)/sizeof(int);
int *evens = result;
int *odds  = result + 5;
thrust::stable_partition_copy(thrust::host, A, A + N, evens, odds, is_even());
// A remains {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
// result is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
// evens points to {2, 4, 6, 8, 10}
// odds points to {1, 3, 5, 7, 9}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code> and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>OutputIterator1</code> and <code>OutputIterator2's</code><code>value&#95;types</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The first element of the sequence to reorder. 
* **`last`** One position past the last element of the sequence to reorder. 
* **`out_true`** The destination of the resulting sequence of elements which satisfy <code>pred</code>. 
* **`out_false`** The destination of the resulting sequence of elements which fail to satisfy <code>pred</code>. 
* **`pred`** A function object which decides to which partition each element of the sequence <code>[first, last)</code> belongs. 

**Preconditions**:
The input ranges shall not overlap with either output range.

**Returns**:
A <code>pair</code> p such that <code>p.first</code> is the end of the output range beginning at <code>out&#95;true</code> and <code>p.second</code> is the end of the output range beginning at <code>out&#95;false</code>.

**See**:
* <a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf">http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf</a>
* <code>partition&#95;copy</code>
* <code>stable&#95;partition</code>

<h3 id="function-stable-partition-copy">
Function <code>thrust::stable&#95;partition&#95;copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>stable_partition_copy</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;OutputIterator1 out_true,</span>
<span>&nbsp;&nbsp;OutputIterator2 out_false,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>stable&#95;partition&#95;copy</code> differs from <code>stable&#95;partition</code> only in that the reordered sequence is written to different output sequences, rather than in place.

<code>stable&#95;partition&#95;copy</code> copies the elements <code>[first, last)</code> based on the function object <code>pred</code>. All of the elements that satisfy <code>pred</code> are copied to the range beginning at <code>out&#95;true</code> and all the elements that fail to satisfy it are copied to the range beginning at <code>out&#95;false</code>.

<code>stable&#95;partition&#95;copy</code> differs from <code>partition&#95;copy</code> in that <code>stable&#95;partition&#95;copy</code> is guaranteed to preserve relative order. That is, if <code>x</code> and <code>y</code> are elements in <code>[first, last)</code>, such that <code>pred(x) == pred(y)</code>, and if <code>x</code> precedes <code>y</code>, then it will still be true after <code>stable&#95;partition&#95;copy</code> that <code>x</code> precedes <code>y</code> in the output.


The following code snippet demonstrates how to use <code>stable&#95;partition&#95;copy</code> to reorder a sequence so that even numbers precede odd numbers.



```cpp
#include <thrust/partition.h>
...
struct is_even
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return (x % 2) == 0;
  }
};
...
int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
int result[10];
const int N = sizeof(A)/sizeof(int);
int *evens = result;
int *odds  = result + 5;
thrust::stable_partition_copy(A, A + N, evens, odds, is_even());
// A remains {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
// result is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
// evens points to {2, 4, 6, 8, 10}
// odds points to {1, 3, 5, 7, 9}
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code> and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>OutputIterator1</code> and <code>OutputIterator2's</code><code>value&#95;types</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first`** The first element of the sequence to reorder. 
* **`last`** One position past the last element of the sequence to reorder. 
* **`out_true`** The destination of the resulting sequence of elements which satisfy <code>pred</code>. 
* **`out_false`** The destination of the resulting sequence of elements which fail to satisfy <code>pred</code>. 
* **`pred`** A function object which decides to which partition each element of the sequence <code>[first, last)</code> belongs. 

**Preconditions**:
The input ranges shall not overlap with either output range.

**Returns**:
A <code>pair</code> p such that <code>p.first</code> is the end of the output range beginning at <code>out&#95;true</code> and <code>p.second</code> is the end of the output range beginning at <code>out&#95;false</code>.

**See**:
* <a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf">http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf</a>
* <code>partition&#95;copy</code>
* <code>stable&#95;partition</code>

<h3 id="function-stable-partition-copy">
Function <code>thrust::stable&#95;partition&#95;copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>stable_partition_copy</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;OutputIterator1 out_true,</span>
<span>&nbsp;&nbsp;OutputIterator2 out_false,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>stable&#95;partition&#95;copy</code> differs from <code>stable&#95;partition</code> only in that the reordered sequence is written to different output sequences, rather than in place.

<code>stable&#95;partition&#95;copy</code> copies the elements <code>[first, last)</code> based on the function object <code>pred</code> which is applied to a range of stencil elements. All of the elements whose corresponding stencil element satisfies <code>pred</code> are copied to the range beginning at <code>out&#95;true</code> and all the elements whose stencil element fails to satisfy it are copied to the range beginning at <code>out&#95;false</code>.

<code>stable&#95;partition&#95;copy</code> differs from <code>partition&#95;copy</code> in that <code>stable&#95;partition&#95;copy</code> is guaranteed to preserve relative order. That is, if <code>x</code> and <code>y</code> are elements in <code>[first, last)</code>, such that <code>pred(x) == pred(y)</code>, and if <code>x</code> precedes <code>y</code>, then it will still be true after <code>stable&#95;partition&#95;copy</code> that <code>x</code> precedes <code>y</code> in the output.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>stable&#95;partition&#95;copy</code> to reorder a sequence so that even numbers precede odd numbers using the <code>thrust::host</code> execution policy for parallelization:



```cpp
#include <thrust/partition.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
...
int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
int S[] = {0, 1, 0, 1, 0, 1, 0, 1, 0,  1};
int result[10];
const int N = sizeof(A)/sizeof(int);
int *evens = result;
int *odds  = result + 5;
thrust::stable_partition_copy(thrust::host, A, A + N, S, evens, odds, thrust::identity<int>());
// A remains {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
// S remains {0, 1, 0, 1, 0, 1, 0, 1, 0,  1}
// result is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
// evens points to {2, 4, 6, 8, 10}
// odds points to {1, 3, 5, 7, 9}
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>OutputIterator1</code> and <code>OutputIterator2's</code><code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The first element of the sequence to reorder. 
* **`last`** One position past the last element of the sequence to reorder. 
* **`stencil`** The beginning of the stencil sequence. 
* **`out_true`** The destination of the resulting sequence of elements which satisfy <code>pred</code>. 
* **`out_false`** The destination of the resulting sequence of elements which fail to satisfy <code>pred</code>. 
* **`pred`** A function object which decides to which partition each element of the sequence <code>[first, last)</code> belongs. 

**Preconditions**:
The input ranges shall not overlap with either output range.

**Returns**:
A <code>pair</code> p such that <code>p.first</code> is the end of the output range beginning at <code>out&#95;true</code> and <code>p.second</code> is the end of the output range beginning at <code>out&#95;false</code>.

**See**:
* <a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf">http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf</a>
* <code>partition&#95;copy</code>
* <code>stable&#95;partition</code>

<h3 id="function-stable-partition-copy">
Function <code>thrust::stable&#95;partition&#95;copy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename OutputIterator1,</span>
<span>&nbsp;&nbsp;typename OutputIterator2,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< OutputIterator1, OutputIterator2 > </span><span><b>stable_partition_copy</b>(InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;OutputIterator1 out_true,</span>
<span>&nbsp;&nbsp;OutputIterator2 out_false,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>stable&#95;partition&#95;copy</code> differs from <code>stable&#95;partition</code> only in that the reordered sequence is written to different output sequences, rather than in place.

<code>stable&#95;partition&#95;copy</code> copies the elements <code>[first, last)</code> based on the function object <code>pred</code> which is applied to a range of stencil elements. All of the elements whose corresponding stencil element satisfies <code>pred</code> are copied to the range beginning at <code>out&#95;true</code> and all the elements whose stencil element fails to satisfy it are copied to the range beginning at <code>out&#95;false</code>.

<code>stable&#95;partition&#95;copy</code> differs from <code>partition&#95;copy</code> in that <code>stable&#95;partition&#95;copy</code> is guaranteed to preserve relative order. That is, if <code>x</code> and <code>y</code> are elements in <code>[first, last)</code>, such that <code>pred(x) == pred(y)</code>, and if <code>x</code> precedes <code>y</code>, then it will still be true after <code>stable&#95;partition&#95;copy</code> that <code>x</code> precedes <code>y</code> in the output.


The following code snippet demonstrates how to use <code>stable&#95;partition&#95;copy</code> to reorder a sequence so that even numbers precede odd numbers.



```cpp
#include <thrust/partition.h>
#include <thrust/functional.h>
...
int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
int S[] = {0, 1, 0, 1, 0, 1, 0, 1, 0,  1};
int result[10];
const int N = sizeof(A)/sizeof(int);
int *evens = result;
int *odds  = result + 5;
thrust::stable_partition_copy(A, A + N, S, evens, odds, thrust::identity<int>());
// A remains {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
// S remains {0, 1, 0, 1, 0, 1, 0, 1, 0,  1}
// result is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
// evens points to {2, 4, 6, 8, 10}
// odds points to {1, 3, 5, 7, 9}
```

**Template Parameters**:
* **`InputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator's</code><code>value&#95;type</code> is convertible to <code>OutputIterator1</code> and <code>OutputIterator2's</code><code>value&#95;types</code>. 
* **`InputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>, and <code>InputIterator2's</code><code>value&#95;type</code> is convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`OutputIterator1`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`OutputIterator2`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`Predicate`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first`** The first element of the sequence to reorder. 
* **`last`** One position past the last element of the sequence to reorder. 
* **`stencil`** The beginning of the stencil sequence. 
* **`out_true`** The destination of the resulting sequence of elements which satisfy <code>pred</code>. 
* **`out_false`** The destination of the resulting sequence of elements which fail to satisfy <code>pred</code>. 
* **`pred`** A function object which decides to which partition each element of the sequence <code>[first, last)</code> belongs. 

**Preconditions**:
The input ranges shall not overlap with either output range.

**Returns**:
A <code>pair</code> p such that <code>p.first</code> is the end of the output range beginning at <code>out&#95;true</code> and <code>p.second</code> is the end of the output range beginning at <code>out&#95;false</code>.

**See**:
* <a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf">http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf</a>
* <code>partition&#95;copy</code>
* <code>stable&#95;partition</code>


