---
title: Scattering
parent: Copying
grand_parent: Algorithms
nav_exclude: false
has_children: true
has_toc: false
---

# Scattering

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__scattering.html#function-scatter">thrust::scatter</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 map,</span>
<span>&nbsp;&nbsp;RandomAccessIterator result);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator&gt;</span>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__scattering.html#function-scatter">thrust::scatter</a></b>(InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 map,</span>
<span>&nbsp;&nbsp;RandomAccessIterator result);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__scattering.html#function-scatter-if">thrust::scatter&#95;if</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 map,</span>
<span>&nbsp;&nbsp;InputIterator3 stencil,</span>
<span>&nbsp;&nbsp;RandomAccessIterator output);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator&gt;</span>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__scattering.html#function-scatter-if">thrust::scatter&#95;if</a></b>(InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 map,</span>
<span>&nbsp;&nbsp;InputIterator3 stencil,</span>
<span>&nbsp;&nbsp;RandomAccessIterator output);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__scattering.html#function-scatter-if">thrust::scatter&#95;if</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 map,</span>
<span>&nbsp;&nbsp;InputIterator3 stencil,</span>
<span>&nbsp;&nbsp;RandomAccessIterator output,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__scattering.html#function-scatter-if">thrust::scatter&#95;if</a></b>(InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 map,</span>
<span>&nbsp;&nbsp;InputIterator3 stencil,</span>
<span>&nbsp;&nbsp;RandomAccessIterator output,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
</code>

## Functions

<h3 id="function-scatter">
Function <code>thrust::scatter</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator&gt;</span>
<span>__host__ __device__ void </span><span><b>scatter</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 map,</span>
<span>&nbsp;&nbsp;RandomAccessIterator result);</span></code>
<code>scatter</code> copies elements from a source range into an output array according to a map. For each iterator <code>i</code> in the range [<code>first</code>, <code>last</code>), the value <code>&#42;i</code> is assigned to <code>output[&#42;(map + (i - first))]</code>. The output iterator must permit random access. If the same index appears more than once in the range <code>[map, map + (last - first))</code>, the result is undefined.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>scatter</code> to reorder a range using the <code>thrust::device</code> execution policy for parallelization:



```cpp
#include <thrust/scatter.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
...
// mark even indices with a 1; odd indices with a 0
int values[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
thrust::device_vector<int> d_values(values, values + 10);

// scatter all even indices into the first half of the
// range, and odd indices vice versa
int map[10]   = {0, 5, 1, 6, 2, 7, 3, 8, 4, 9};
thrust::device_vector<int> d_map(map, map + 10);

thrust::device_vector<int> d_output(10);
thrust::scatter(thrust::device,
                d_values.begin(), d_values.end(),
                d_map.begin(), d_output.begin());
// d_output is now {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}
```

**Note**:
<code>scatter</code> is the inverse of thrust::gather. 

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator1's</code><code>value&#95;type</code> must be convertible to <code>RandomAccessIterator's</code><code>value&#95;type</code>. 
* **`InputIterator2`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> must be convertible to <code>RandomAccessIterator's</code><code>difference&#95;type</code>. 
* **`RandomAccessIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access iterator</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** Beginning of the sequence of values to scatter. 
* **`last`** End of the sequence of values to scatter. 
* **`map`** Beginning of the sequence of output indices. 
* **`result`** Destination of the source elements.

**Preconditions**:
* The iterator <code>result + i</code> shall not refer to any element referenced by any iterator <code>j</code> in the range <code>[first,last)</code> for all iterators <code>i</code> in the range <code>[map,map + (last - first))</code>.
* The iterator <code>result + i</code> shall not refer to any element referenced by any iterator <code>j</code> in the range <code>[map,map + (last - first))</code> for all iterators <code>i</code> in the range <code>[map,map + (last - first))</code>.
* The expression <code>result[&#42;i]</code> shall be valid for all iterators in the range <code>[map,map + (last - first))</code>.

<h3 id="function-scatter">
Function <code>thrust::scatter</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator&gt;</span>
<span>void </span><span><b>scatter</b>(InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 map,</span>
<span>&nbsp;&nbsp;RandomAccessIterator result);</span></code>
<code>scatter</code> copies elements from a source range into an output array according to a map. For each iterator <code>i</code> in the range [<code>first</code>, <code>last</code>), the value <code>&#42;i</code> is assigned to <code>output[&#42;(map + (i - first))]</code>. The output iterator must permit random access. If the same index appears more than once in the range <code>[map, map + (last - first))</code>, the result is undefined.


The following code snippet demonstrates how to use <code>scatter</code> to reorder a range.



```cpp
#include <thrust/scatter.h>
#include <thrust/device_vector.h>
...
// mark even indices with a 1; odd indices with a 0
int values[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
thrust::device_vector<int> d_values(values, values + 10);

// scatter all even indices into the first half of the
// range, and odd indices vice versa
int map[10]   = {0, 5, 1, 6, 2, 7, 3, 8, 4, 9};
thrust::device_vector<int> d_map(map, map + 10);

thrust::device_vector<int> d_output(10);
thrust::scatter(d_values.begin(), d_values.end(),
                d_map.begin(), d_output.begin());
// d_output is now {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}
```

**Note**:
<code>scatter</code> is the inverse of thrust::gather. 

**Template Parameters**:
* **`InputIterator1`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator1's</code><code>value&#95;type</code> must be convertible to <code>RandomAccessIterator's</code><code>value&#95;type</code>. 
* **`InputIterator2`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> must be convertible to <code>RandomAccessIterator's</code><code>difference&#95;type</code>. 
* **`RandomAccessIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access iterator</a>.

**Function Parameters**:
* **`first`** Beginning of the sequence of values to scatter. 
* **`last`** End of the sequence of values to scatter. 
* **`map`** Beginning of the sequence of output indices. 
* **`result`** Destination of the source elements.

**Preconditions**:
* The iterator <code>result + i</code> shall not refer to any element referenced by any iterator <code>j</code> in the range <code>[first,last)</code> for all iterators <code>i</code> in the range <code>[map,map + (last - first))</code>.
* The iterator <code>result + i</code> shall not refer to any element referenced by any iterator <code>j</code> in the range <code>[map,map + (last - first))</code> for all iterators <code>i</code> in the range <code>[map,map + (last - first))</code>.
* The expression <code>result[&#42;i]</code> shall be valid for all iterators in the range <code>[map,map + (last - first))</code>.

<h3 id="function-scatter-if">
Function <code>thrust::scatter&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator&gt;</span>
<span>__host__ __device__ void </span><span><b>scatter_if</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 map,</span>
<span>&nbsp;&nbsp;InputIterator3 stencil,</span>
<span>&nbsp;&nbsp;RandomAccessIterator output);</span></code>
<code>scatter&#95;if</code> conditionally copies elements from a source range into an output array according to a map. For each iterator <code>i</code> in the range <code>[first, last)</code> such that <code>&#42;(stencil + (i - first))</code> is true, the value <code>&#42;i</code> is assigned to <code>output[&#42;(map + (i - first))]</code>. The output iterator must permit random access. If the same index appears more than once in the range <code>[map, map + (last - first))</code> the result is undefined.

The algorithm's execution is parallelized as determined by <code>exec</code>.



```cpp
#include <thrust/scatter.h>
#include <thrust/execution_policy.h>
...
int V[8] = {10, 20, 30, 40, 50, 60, 70, 80};
int M[8] = {0, 5, 1, 6, 2, 7, 3, 4};
int S[8] = {1, 0, 1, 0, 1, 0, 1, 0};
int D[8] = {0, 0, 0, 0, 0, 0, 0, 0};

thrust::scatter_if(thrust::host, V, V + 8, M, S, D);

// D contains [10, 30, 50, 70, 0, 0, 0, 0];
```

**Note**:
<code>scatter&#95;if</code> is the inverse of thrust::gather_if. 

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator1's</code><code>value&#95;type</code> must be convertible to <code>RandomAccessIterator's</code><code>value&#95;type</code>. 
* **`InputIterator2`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> must be convertible to <code>RandomAccessIterator's</code><code>difference&#95;type</code>. 
* **`InputIterator3`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator3's</code><code>value&#95;type</code> must be convertible to <code>bool</code>. 
* **`RandomAccessIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access iterator</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** Beginning of the sequence of values to scatter. 
* **`last`** End of the sequence of values to scatter. 
* **`map`** Beginning of the sequence of output indices. 
* **`stencil`** Beginning of the sequence of predicate values. 
* **`output`** Beginning of the destination range.

**Preconditions**:
* The iterator <code>result + i</code> shall not refer to any element referenced by any iterator <code>j</code> in the range <code>[first,last)</code> for all iterators <code>i</code> in the range <code>[map,map + (last - first))</code>.
* The iterator <code>result + i</code> shall not refer to any element referenced by any iterator <code>j</code> in the range <code>[map,map + (last - first))</code> for all iterators <code>i</code> in the range <code>[map,map + (last - first))</code>.
* The iterator <code>result + i</code> shall not refer to any element referenced by any iterator <code>j</code> in the range <code>[stencil,stencil + (last - first))</code> for all iterators <code>i</code> in the range <code>[map,map + (last - first))</code>.
* The expression <code>result[&#42;i]</code> shall be valid for all iterators <code>i</code> in the range <code>[map,map + (last - first))</code> for which the following condition holds: <code>&#42;(stencil + i) != false</code>.

<h3 id="function-scatter-if">
Function <code>thrust::scatter&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator&gt;</span>
<span>void </span><span><b>scatter_if</b>(InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 map,</span>
<span>&nbsp;&nbsp;InputIterator3 stencil,</span>
<span>&nbsp;&nbsp;RandomAccessIterator output);</span></code>
<code>scatter&#95;if</code> conditionally copies elements from a source range into an output array according to a map. For each iterator <code>i</code> in the range <code>[first, last)</code> such that <code>&#42;(stencil + (i - first))</code> is true, the value <code>&#42;i</code> is assigned to <code>output[&#42;(map + (i - first))]</code>. The output iterator must permit random access. If the same index appears more than once in the range <code>[map, map + (last - first))</code> the result is undefined.



```cpp
#include <thrust/scatter.h>
...
int V[8] = {10, 20, 30, 40, 50, 60, 70, 80};
int M[8] = {0, 5, 1, 6, 2, 7, 3, 4};
int S[8] = {1, 0, 1, 0, 1, 0, 1, 0};
int D[8] = {0, 0, 0, 0, 0, 0, 0, 0};

thrust::scatter_if(V, V + 8, M, S, D);

// D contains [10, 30, 50, 70, 0, 0, 0, 0];
```

**Note**:
<code>scatter&#95;if</code> is the inverse of thrust::gather_if. 

**Template Parameters**:
* **`InputIterator1`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator1's</code><code>value&#95;type</code> must be convertible to <code>RandomAccessIterator's</code><code>value&#95;type</code>. 
* **`InputIterator2`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> must be convertible to <code>RandomAccessIterator's</code><code>difference&#95;type</code>. 
* **`InputIterator3`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator3's</code><code>value&#95;type</code> must be convertible to <code>bool</code>. 
* **`RandomAccessIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access iterator</a>.

**Function Parameters**:
* **`first`** Beginning of the sequence of values to scatter. 
* **`last`** End of the sequence of values to scatter. 
* **`map`** Beginning of the sequence of output indices. 
* **`stencil`** Beginning of the sequence of predicate values. 
* **`output`** Beginning of the destination range.

**Preconditions**:
* The iterator <code>result + i</code> shall not refer to any element referenced by any iterator <code>j</code> in the range <code>[first,last)</code> for all iterators <code>i</code> in the range <code>[map,map + (last - first))</code>.
* The iterator <code>result + i</code> shall not refer to any element referenced by any iterator <code>j</code> in the range <code>[map,map + (last - first))</code> for all iterators <code>i</code> in the range <code>[map,map + (last - first))</code>.
* The iterator <code>result + i</code> shall not refer to any element referenced by any iterator <code>j</code> in the range <code>[stencil,stencil + (last - first))</code> for all iterators <code>i</code> in the range <code>[map,map + (last - first))</code>.
* The expression <code>result[&#42;i]</code> shall be valid for all iterators <code>i</code> in the range <code>[map,map + (last - first))</code> for which the following condition holds: <code>&#42;(stencil + i) != false</code>.

<h3 id="function-scatter-if">
Function <code>thrust::scatter&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ void </span><span><b>scatter_if</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 map,</span>
<span>&nbsp;&nbsp;InputIterator3 stencil,</span>
<span>&nbsp;&nbsp;RandomAccessIterator output,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>scatter&#95;if</code> conditionally copies elements from a source range into an output array according to a map. For each iterator <code>i</code> in the range <code>[first, last)</code> such that <code>pred(&#42;(stencil + (i - first)))</code> is <code>true</code>, the value <code>&#42;i</code> is assigned to <code>output[&#42;(map + (i - first))]</code>. The output iterator must permit random access. If the same index appears more than once in the range <code>[map, map + (last - first))</code> the result is undefined.

The algorithm's execution is parallelized as determined by <code>exec</code>.



```cpp
#include <thrust/scatter.h>
#include <thrust/execution_policy.h>

struct is_even
{
  __host__ __device__
  bool operator()(int x)
  {
    return (x % 2) == 0;
  }
};

...

int V[8] = {10, 20, 30, 40, 50, 60, 70, 80};
int M[8] = {0, 5, 1, 6, 2, 7, 3, 4};
int S[8] = {2, 1, 2, 1, 2, 1, 2, 1};
int D[8] = {0, 0, 0, 0, 0, 0, 0, 0};

is_even pred;
thrust::scatter_if(thrust::host, V, V + 8, M, S, D, pred);

// D contains [10, 30, 50, 70, 0, 0, 0, 0];
```

**Note**:
<code>scatter&#95;if</code> is the inverse of thrust::gather_if. 

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator1`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator1's</code><code>value&#95;type</code> must be convertible to <code>RandomAccessIterator's</code><code>value&#95;type</code>. 
* **`InputIterator2`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> must be convertible to <code>RandomAccessIterator's</code><code>difference&#95;type</code>. 
* **`InputIterator3`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator3's</code><code>value&#95;type</code> must be convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`RandomAccessIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access iterator</a>. 
* **`Predicate`** must be a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** Beginning of the sequence of values to scatter. 
* **`last`** End of the sequence of values to scatter. 
* **`map`** Beginning of the sequence of output indices. 
* **`stencil`** Beginning of the sequence of predicate values. 
* **`output`** Beginning of the destination range. 
* **`pred`** Predicate to apply to the stencil values.

**Preconditions**:
* The iterator <code>result + i</code> shall not refer to any element referenced by any iterator <code>j</code> in the range <code>[first,last)</code> for all iterators <code>i</code> in the range <code>[map,map + (last - first))</code>.
* The iterator <code>result + i</code> shall not refer to any element referenced by any iterator <code>j</code> in the range <code>[map,map + (last - first))</code> for all iterators <code>i</code> in the range <code>[map,map + (last - first))</code>.
* The iterator <code>result + i</code> shall not refer to any element referenced by any iterator <code>j</code> in the range <code>[stencil,stencil + (last - first))</code> for all iterators <code>i</code> in the range <code>[map,map + (last - first))</code>.
* The expression <code>result[&#42;i]</code> shall be valid for all iterators <code>i</code> in the range <code>[map,map + (last - first))</code> for which the following condition holds: <code>pred(&#42;(stencil + i)) != false</code>.

<h3 id="function-scatter-if">
Function <code>thrust::scatter&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator1,</span>
<span>&nbsp;&nbsp;typename InputIterator2,</span>
<span>&nbsp;&nbsp;typename InputIterator3,</span>
<span>&nbsp;&nbsp;typename RandomAccessIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>void </span><span><b>scatter_if</b>(InputIterator1 first,</span>
<span>&nbsp;&nbsp;InputIterator1 last,</span>
<span>&nbsp;&nbsp;InputIterator2 map,</span>
<span>&nbsp;&nbsp;InputIterator3 stencil,</span>
<span>&nbsp;&nbsp;RandomAccessIterator output,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>scatter&#95;if</code> conditionally copies elements from a source range into an output array according to a map. For each iterator <code>i</code> in the range <code>[first, last)</code> such that <code>pred(&#42;(stencil + (i - first)))</code> is <code>true</code>, the value <code>&#42;i</code> is assigned to <code>output[&#42;(map + (i - first))]</code>. The output iterator must permit random access. If the same index appears more than once in the range <code>[map, map + (last - first))</code> the result is undefined.



```cpp
#include <thrust/scatter.h>

struct is_even
{
  __host__ __device__
  bool operator()(int x)
  {
    return (x % 2) == 0;
  }
};

...

int V[8] = {10, 20, 30, 40, 50, 60, 70, 80};
int M[8] = {0, 5, 1, 6, 2, 7, 3, 4};
int S[8] = {2, 1, 2, 1, 2, 1, 2, 1};
int D[8] = {0, 0, 0, 0, 0, 0, 0, 0};

is_even pred;
thrust::scatter_if(V, V + 8, M, S, D, pred);

// D contains [10, 30, 50, 70, 0, 0, 0, 0];
```

**Note**:
<code>scatter&#95;if</code> is the inverse of thrust::gather_if. 

**Template Parameters**:
* **`InputIterator1`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator1's</code><code>value&#95;type</code> must be convertible to <code>RandomAccessIterator's</code><code>value&#95;type</code>. 
* **`InputIterator2`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator2's</code><code>value&#95;type</code> must be convertible to <code>RandomAccessIterator's</code><code>difference&#95;type</code>. 
* **`InputIterator3`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator3's</code><code>value&#95;type</code> must be convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`RandomAccessIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/random_access_iterator">Random Access iterator</a>. 
* **`Predicate`** must be a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first`** Beginning of the sequence of values to scatter. 
* **`last`** End of the sequence of values to scatter. 
* **`map`** Beginning of the sequence of output indices. 
* **`stencil`** Beginning of the sequence of predicate values. 
* **`output`** Beginning of the destination range. 
* **`pred`** Predicate to apply to the stencil values.

**Preconditions**:
* The iterator <code>result + i</code> shall not refer to any element referenced by any iterator <code>j</code> in the range <code>[first,last)</code> for all iterators <code>i</code> in the range <code>[map,map + (last - first))</code>.
* The iterator <code>result + i</code> shall not refer to any element referenced by any iterator <code>j</code> in the range <code>[map,map + (last - first))</code> for all iterators <code>i</code> in the range <code>[map,map + (last - first))</code>.
* The iterator <code>result + i</code> shall not refer to any element referenced by any iterator <code>j</code> in the range <code>[stencil,stencil + (last - first))</code> for all iterators <code>i</code> in the range <code>[map,map + (last - first))</code>.
* The expression <code>result[&#42;i]</code> shall be valid for all iterators <code>i</code> in the range <code>[map,map + (last - first))</code> for which the following condition holds: <code>pred(&#42;(stencil + i)) != false</code>.


