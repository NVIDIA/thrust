---
title: Gathering
parent: Copying
grand_parent: Algorithms
nav_exclude: false
has_children: true
has_toc: false
---

# Gathering

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__gathering.html#function-gather">thrust::gather</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator map_first,</span>
<span>&nbsp;&nbsp;InputIterator map_last,</span>
<span>&nbsp;&nbsp;RandomAccessIterator input_first,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__gathering.html#function-gather">thrust::gather</a></b>(InputIterator map_first,</span>
<span>&nbsp;&nbsp;InputIterator map_last,</span>
<span>&nbsp;&nbsp;RandomAccessIterator input_first,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__gathering.html#function-gather-if">thrust::gather&#95;if</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 map_first,</span>
<span>&nbsp;&nbsp;InputIterator1 map_last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;RandomAccessIterator input_first,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__gathering.html#function-gather-if">thrust::gather&#95;if</a></b>(InputIterator1 map_first,</span>
<span>&nbsp;&nbsp;InputIterator1 map_last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;RandomAccessIterator input_first,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__gathering.html#function-gather-if">thrust::gather&#95;if</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 map_first,</span>
<span>&nbsp;&nbsp;InputIterator1 map_last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;RandomAccessIterator input_first,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>OutputIterator </span><span><b><a href="{{ site.baseurl }}/api/groups/group__gathering.html#function-gather-if">thrust::gather&#95;if</a></b>(InputIterator1 map_first,</span>
<span>&nbsp;&nbsp;InputIterator1 map_last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;RandomAccessIterator input_first,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
</code>

## Functions

<h3 id="function-gather">
Function <code>thrust::gather</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>gather</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator map_first,</span>
<span>&nbsp;&nbsp;InputIterator map_last,</span>
<span>&nbsp;&nbsp;RandomAccessIterator input_first,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>gather</code> copies elements from a source array into a destination range according to a map. For each input iterator <code>i</code> in the range <code>[map&#95;first, map&#95;last)</code>, the value <code>input&#95;first[&#42;i]</code> is assigned to <code>&#42;(result + (i - map&#95;first))</code>. <code>RandomAccessIterator</code> must permit random access.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>gather</code> to reorder a range using the <code>thrust::device</code> execution policy for parallelization:



```cpp
#include <thrust/gather.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
...
// mark even indices with a 1; odd indices with a 0
int values[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
thrust::device_vector<int> d_values(values, values + 10);

// gather all even indices into the first half of the range
// and odd indices to the last half of the range
int map[10]   = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};
thrust::device_vector<int> d_map(map, map + 10);

thrust::device_vector<int> d_output(10);
thrust::gather(thrust::device,
               d_map.begin(), d_map.end(),
               d_values.begin(),
               d_output.begin());
// d_output is now {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}
```

**Remark**:
<code>gather</code> is the inverse of thrust::scatter.

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> must be convertible to <code>RandomAccessIterator's</code><code>difference&#95;type</code>. 
* **`RandomAccessIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a> and <code>RandomAccessIterator's</code><code>value&#95;type</code> must be convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`map_first`** Beginning of the range of gather locations. 
* **`map_last`** End of the range of gather locations. 
* **`input_first`** Beginning of the source range. 
* **`result`** Beginning of the destination range.

**Preconditions**:
* The range <code>[map&#95;first, map&#95;last)</code> shall not overlap the range <code>[result, result + (map&#95;last - map&#95;first))</code>. 
* The input data shall not overlap the range <code>[result, result + (map&#95;last - map&#95;first))</code>.

<h3 id="function-gather">
Function <code>thrust::gather</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b>gather</b>(InputIterator map_first,</span>
<span>&nbsp;&nbsp;InputIterator map_last,</span>
<span>&nbsp;&nbsp;RandomAccessIterator input_first,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>gather</code> copies elements from a source array into a destination range according to a map. For each input iterator <code>i</code> in the range <code>[map&#95;first, map&#95;last)</code>, the value <code>input&#95;first[&#42;i]</code> is assigned to <code>&#42;(result + (i - map&#95;first))</code>. <code>RandomAccessIterator</code> must permit random access.


The following code snippet demonstrates how to use <code>gather</code> to reorder a range.



```cpp
#include <thrust/gather.h>
#include <thrust/device_vector.h>
...
// mark even indices with a 1; odd indices with a 0
int values[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
thrust::device_vector<int> d_values(values, values + 10);

// gather all even indices into the first half of the range
// and odd indices to the last half of the range
int map[10]   = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};
thrust::device_vector<int> d_map(map, map + 10);

thrust::device_vector<int> d_output(10);
thrust::gather(d_map.begin(), d_map.end(),
               d_values.begin(),
               d_output.begin());
// d_output is now {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}
```

**Remark**:
<code>gather</code> is the inverse of thrust::scatter.

**Template Parameters**:
* **`InputIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> must be convertible to <code>RandomAccessIterator's</code><code>difference&#95;type</code>. 
* **`RandomAccessIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access Iterator</a> and <code>RandomAccessIterator's</code><code>value&#95;type</code> must be convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`map_first`** Beginning of the range of gather locations. 
* **`map_last`** End of the range of gather locations. 
* **`input_first`** Beginning of the source range. 
* **`result`** Beginning of the destination range.

**Preconditions**:
* The range <code>[map&#95;first, map&#95;last)</code> shall not overlap the range <code>[result, result + (map&#95;last - map&#95;first))</code>. 
* The input data shall not overlap the range <code>[result, result + (map&#95;last - map&#95;first))</code>.

<h3 id="function-gather-if">
Function <code>thrust::gather&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>gather_if</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 map_first,</span>
<span>&nbsp;&nbsp;InputIterator1 map_last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;RandomAccessIterator input_first,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>gather&#95;if</code> conditionally copies elements from a source array into a destination range according to a map. For each input iterator <code>i</code> in the range <code>[map&#95;first, map&#95;last)</code>, such that the value of <code>&#42;(stencil + (i - map&#95;first))</code> is <code>true</code>, the value <code>input&#95;first[&#42;i]</code> is assigned to <code>&#42;(result + (i - map&#95;first))</code>. <code>RandomAccessIterator</code> must permit random access.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>gather&#95;if</code> to gather selected values from an input range using the <code>thrust::device</code> execution policy:



```cpp
#include <thrust/gather.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
...

int values[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
thrust::device_vector<int> d_values(values, values + 10);

// select elements at even-indexed locations
int stencil[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
thrust::device_vector<int> d_stencil(stencil, stencil + 10);

// map all even indices into the first half of the range
// and odd indices to the last half of the range
int map[10]   = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};
thrust::device_vector<int> d_map(map, map + 10);

thrust::device_vector<int> d_output(10, 7);
thrust::gather_if(thrust::device,
                  d_map.begin(), d_map.end(),
                  d_stencil.begin(),
                  d_values.begin(),
                  d_output.begin());
// d_output is now {0, 7, 4, 7, 8, 7, 3, 7, 7, 7}
```

**Remark**:
<code>gather&#95;if</code> is the inverse of <code>scatter&#95;if</code>.

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator1's</code><code>value&#95;type</code> must be convertible to <code>RandomAccessIterator's</code><code>difference&#95;type</code>. 
* **`InputIterator2`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> must be convertible to <code>bool</code>. 
* **`RandomAccessIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access iterator</a> and <code>RandomAccessIterator's</code><code>value&#95;type</code> must be convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`map_first`** Beginning of the range of gather locations. 
* **`map_last`** End of the range of gather locations. 
* **`stencil`** Beginning of the range of predicate values. 
* **`input_first`** Beginning of the source range. 
* **`result`** Beginning of the destination range.

**Preconditions**:
* The range <code>[map&#95;first, map&#95;last)</code> shall not overlap the range <code>[result, result + (map&#95;last - map&#95;first))</code>. 
* The range <code>[stencil, stencil + (map&#95;last - map&#95;first))</code> shall not overlap the range <code>[result, result + (map&#95;last - map&#95;first))</code>. 
* The input data shall not overlap the range <code>[result, result + (map&#95;last - map&#95;first))</code>.

<h3 id="function-gather-if">
Function <code>thrust::gather&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>OutputIterator </span><span><b>gather_if</b>(InputIterator1 map_first,</span>
<span>&nbsp;&nbsp;InputIterator1 map_last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;RandomAccessIterator input_first,</span>
<span>&nbsp;&nbsp;OutputIterator result);</span></code>
<code>gather&#95;if</code> conditionally copies elements from a source array into a destination range according to a map. For each input iterator <code>i</code> in the range <code>[map&#95;first, map&#95;last)</code>, such that the value of <code>&#42;(stencil + (i - map&#95;first))</code> is <code>true</code>, the value <code>input&#95;first[&#42;i]</code> is assigned to <code>&#42;(result + (i - map&#95;first))</code>. <code>RandomAccessIterator</code> must permit random access.


The following code snippet demonstrates how to use <code>gather&#95;if</code> to gather selected values from an input range.



```cpp
#include <thrust/gather.h>
#include <thrust/device_vector.h>
...

int values[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
thrust::device_vector<int> d_values(values, values + 10);

// select elements at even-indexed locations
int stencil[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
thrust::device_vector<int> d_stencil(stencil, stencil + 10);

// map all even indices into the first half of the range
// and odd indices to the last half of the range
int map[10]   = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};
thrust::device_vector<int> d_map(map, map + 10);

thrust::device_vector<int> d_output(10, 7);
thrust::gather_if(d_map.begin(), d_map.end(),
                  d_stencil.begin(),
                  d_values.begin(),
                  d_output.begin());
// d_output is now {0, 7, 4, 7, 8, 7, 3, 7, 7, 7}
```

**Remark**:
<code>gather&#95;if</code> is the inverse of <code>scatter&#95;if</code>.

**Template Parameters**:
* **`InputIterator1`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator1's</code><code>value&#95;type</code> must be convertible to <code>RandomAccessIterator's</code><code>difference&#95;type</code>. 
* **`InputIterator2`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> must be convertible to <code>bool</code>. 
* **`RandomAccessIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access iterator</a> and <code>RandomAccessIterator's</code><code>value&#95;type</code> must be convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.

**Function Parameters**:
* **`map_first`** Beginning of the range of gather locations. 
* **`map_last`** End of the range of gather locations. 
* **`stencil`** Beginning of the range of predicate values. 
* **`input_first`** Beginning of the source range. 
* **`result`** Beginning of the destination range.

**Preconditions**:
* The range <code>[map&#95;first, map&#95;last)</code> shall not overlap the range <code>[result, result + (map&#95;last - map&#95;first))</code>. 
* The range <code>[stencil, stencil + (map&#95;last - map&#95;first))</code> shall not overlap the range <code>[result, result + (map&#95;last - map&#95;first))</code>. 
* The input data shall not overlap the range <code>[result, result + (map&#95;last - map&#95;first))</code>.

<h3 id="function-gather-if">
Function <code>thrust::gather&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ OutputIterator </span><span><b>gather_if</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 map_first,</span>
<span>&nbsp;&nbsp;InputIterator1 map_last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;RandomAccessIterator input_first,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>gather&#95;if</code> conditionally copies elements from a source array into a destination range according to a map. For each input iterator <code>i</code> in the range <code>[map&#95;first, map&#95;last)</code> such that the value of <code>pred(&#42;(stencil + (i - map&#95;first)))</code> is <code>true</code>, the value <code>input&#95;first[&#42;i]</code> is assigned to <code>&#42;(result + (i - map&#95;first))</code>. <code>RandomAccessIterator</code> must permit random access.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>gather&#95;if</code> to gather selected values from an input range based on an arbitrary selection function using the <code>thrust::device</code> execution policy for parallelization:



```cpp
#include <thrust/gather.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

struct is_even
{
  __host__ __device__
  bool operator()(const int x)
  {
    return (x % 2) == 0;
  }
};
...

int values[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
thrust::device_vector<int> d_values(values, values + 10);

// we will select an element when our stencil is even
int stencil[10] = {0, 3, 4, 1, 4, 1, 2, 7, 8, 9};
thrust::device_vector<int> d_stencil(stencil, stencil + 10);

// map all even indices into the first half of the range
// and odd indices to the last half of the range
int map[10]   = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};
thrust::device_vector<int> d_map(map, map + 10);

thrust::device_vector<int> d_output(10, 7);
thrust::gather_if(thrust::device,
                  d_map.begin(), d_map.end(),
                  d_stencil.begin(),
                  d_values.begin(),
                  d_output.begin(),
                  is_even());
// d_output is now {0, 7, 4, 7, 8, 7, 3, 7, 7, 7}
```

**Remark**:
<code>gather&#95;if</code> is the inverse of <code>scatter&#95;if</code>.

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator1's</code><code>value&#95;type</code> must be convertible to <code>RandomAccessIterator's</code><code>difference&#95;type</code>. 
* **`InputIterator2`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> must be convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`RandomAccessIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access iterator</a> and <code>RandomAccessIterator's</code><code>value&#95;type</code> must be convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`Predicate`** must be a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`map_first`** Beginning of the range of gather locations. 
* **`map_last`** End of the range of gather locations. 
* **`stencil`** Beginning of the range of predicate values. 
* **`input_first`** Beginning of the source range. 
* **`result`** Beginning of the destination range. 
* **`pred`** Predicate to apply to the stencil values.

**Preconditions**:
* The range <code>[map&#95;first, map&#95;last)</code> shall not overlap the range <code>[result, result + (map&#95;last - map&#95;first))</code>. 
* The range <code>[stencil, stencil + (map&#95;last - map&#95;first))</code> shall not overlap the range <code>[result, result + (map&#95;last - map&#95;first))</code>. 
* The input data shall not overlap the range <code>[result, result + (map&#95;last - map&#95;first))</code>.

<h3 id="function-gather-if">
Function <code>thrust::gather&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator,</span>
<span>&nbsp;&nbsp;typename OutputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>OutputIterator </span><span><b>gather_if</b>(InputIterator1 map_first,</span>
<span>&nbsp;&nbsp;InputIterator1 map_last,</span>
<span>&nbsp;&nbsp;InputIterator2 stencil,</span>
<span>&nbsp;&nbsp;RandomAccessIterator input_first,</span>
<span>&nbsp;&nbsp;OutputIterator result,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>gather&#95;if</code> conditionally copies elements from a source array into a destination range according to a map. For each input iterator <code>i</code> in the range <code>[map&#95;first, map&#95;last)</code> such that the value of <code>pred(&#42;(stencil + (i - map&#95;first)))</code> is <code>true</code>, the value <code>input&#95;first[&#42;i]</code> is assigned to <code>&#42;(result + (i - map&#95;first))</code>. <code>RandomAccessIterator</code> must permit random access.


The following code snippet demonstrates how to use <code>gather&#95;if</code> to gather selected values from an input range based on an arbitrary selection function.



```cpp
#include <thrust/gather.h>
#include <thrust/device_vector.h>

struct is_even
{
  __host__ __device__
  bool operator()(const int x)
  {
    return (x % 2) == 0;
  }
};
...

int values[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
thrust::device_vector<int> d_values(values, values + 10);

// we will select an element when our stencil is even
int stencil[10] = {0, 3, 4, 1, 4, 1, 2, 7, 8, 9};
thrust::device_vector<int> d_stencil(stencil, stencil + 10);

// map all even indices into the first half of the range
// and odd indices to the last half of the range
int map[10]   = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};
thrust::device_vector<int> d_map(map, map + 10);

thrust::device_vector<int> d_output(10, 7);
thrust::gather_if(d_map.begin(), d_map.end(),
                  d_stencil.begin(),
                  d_values.begin(),
                  d_output.begin(),
                  is_even());
// d_output is now {0, 7, 4, 7, 8, 7, 3, 7, 7, 7}
```

**Remark**:
<code>gather&#95;if</code> is the inverse of <code>scatter&#95;if</code>.

**Template Parameters**:
* **`InputIterator1`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator1's</code><code>value&#95;type</code> must be convertible to <code>RandomAccessIterator's</code><code>difference&#95;type</code>. 
* **`InputIterator2`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> must be convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`RandomAccessIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access iterator</a> and <code>RandomAccessIterator's</code><code>value&#95;type</code> must be convertible to <code>OutputIterator's</code><code>value&#95;type</code>. 
* **`OutputIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>. 
* **`Predicate`** must be a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`map_first`** Beginning of the range of gather locations. 
* **`map_last`** End of the range of gather locations. 
* **`stencil`** Beginning of the range of predicate values. 
* **`input_first`** Beginning of the source range. 
* **`result`** Beginning of the destination range. 
* **`pred`** Predicate to apply to the stencil values.

**Preconditions**:
* The range <code>[map&#95;first, map&#95;last)</code> shall not overlap the range <code>[result, result + (map&#95;last - map&#95;first))</code>. 
* The range <code>[stencil, stencil + (map&#95;last - map&#95;first))</code> shall not overlap the range <code>[result, result + (map&#95;last - map&#95;first))</code>. 
* The input data shall not overlap the range <code>[result, result + (map&#95;last - map&#95;first))</code>.


