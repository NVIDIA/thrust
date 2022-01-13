---
title: Counting
parent: Reductions
grand_parent: Algorithms
nav_exclude: false
has_children: true
has_toc: false
---

# Counting

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename EqualityComparable&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">thrust::iterator_traits</a>< InputIterator >::difference_type </span><span><b><a href="{{ site.baseurl }}/api/groups/group__counting.html#function-count">thrust::count</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;const EqualityComparable & value);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename EqualityComparable&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">thrust::iterator_traits</a>< InputIterator >::difference_type </span><span><b><a href="{{ site.baseurl }}/api/groups/group__counting.html#function-count">thrust::count</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;const EqualityComparable & value);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">thrust::iterator_traits</a>< InputIterator >::difference_type </span><span><b><a href="{{ site.baseurl }}/api/groups/group__counting.html#function-count-if">thrust::count&#95;if</a></b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
<br>
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">thrust::iterator_traits</a>< InputIterator >::difference_type </span><span><b><a href="{{ site.baseurl }}/api/groups/group__counting.html#function-count-if">thrust::count&#95;if</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span>
</code>

## Functions

<h3 id="function-count">
Function <code>thrust::count</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename EqualityComparable&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">thrust::iterator_traits</a>< InputIterator >::difference_type </span><span><b>count</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;const EqualityComparable & value);</span></code>
<code>count</code> finds the number of elements in <code>[first,last)</code> that are equal to <code>value</code>. More precisely, <code>count</code> returns the number of iterators <code>i</code> in <code>[first, last)</code> such that <code>&#42;i == value</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>count</code> to count the number of instances in a range of a value of interest using the <code>thrust::device</code> execution policy:



```cpp
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
...
// put 3 1s in a device_vector
thrust::device_vector<int> vec(5,0);
vec[1] = 1;
vec[3] = 1;
vec[4] = 1;

// count the 1s
int result = thrust::count(thrust::device, vec.begin(), vec.end(), 1);
// result == 3
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> must be a model of must be a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>. 
* **`EqualityComparable`** must be a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a> and can be compared for equality with <code>InputIterator's</code><code>value&#95;type</code>

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`value`** The value to be counted. 

**Returns**:
The number of elements equal to <code>value</code>.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/count">https://en.cppreference.com/w/cpp/algorithm/count</a>

<h3 id="function-count">
Function <code>thrust::count</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename EqualityComparable&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">thrust::iterator_traits</a>< InputIterator >::difference_type </span><span><b>count</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;const EqualityComparable & value);</span></code>
<code>count</code> finds the number of elements in <code>[first,last)</code> that are equal to <code>value</code>. More precisely, <code>count</code> returns the number of iterators <code>i</code> in <code>[first, last)</code> such that <code>&#42;i == value</code>.


The following code snippet demonstrates how to use <code>count</code> to count the number of instances in a range of a value of interest. 

```cpp
#include <thrust/count.h>
#include <thrust/device_vector.h>
...
// put 3 1s in a device_vector
thrust::device_vector<int> vec(5,0);
vec[1] = 1;
vec[3] = 1;
vec[4] = 1;

// count the 1s
int result = thrust::count(vec.begin(), vec.end(), 1);
// result == 3
```

**Template Parameters**:
* **`InputIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> must be a model of must be a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>. 
* **`EqualityComparable`** must be a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a> and can be compared for equality with <code>InputIterator's</code><code>value&#95;type</code>

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`value`** The value to be counted. 

**Returns**:
The number of elements equal to <code>value</code>.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/count">https://en.cppreference.com/w/cpp/algorithm/count</a>

<h3 id="function-count-if">
Function <code>thrust::count&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">thrust::iterator_traits</a>< InputIterator >::difference_type </span><span><b>count_if</b>(const thrust::detail::execution_policy_base< DerivedPolicy > & exec,</span>
<span>&nbsp;&nbsp;InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>count&#95;if</code> finds the number of elements in <code>[first,last)</code> for which a predicate is <code>true</code>. More precisely, <code>count&#95;if</code> returns the number of iterators <code>i</code> in <code>[first, last)</code> such that <code>pred(&#42;i) == true</code>.

The algorithm's execution is parallelized as determined by <code>exec</code>.


The following code snippet demonstrates how to use <code>count</code> to count the number of odd numbers in a range using the <code>thrust::device</code> execution policy:



```cpp
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
...
struct is_odd
{
  __host__ __device__
  bool operator()(int &x)
  {
    return x & 1;
  }
};
...
// fill a device_vector with even & odd numbers
thrust::device_vector<int> vec(5);
vec[0] = 0;
vec[1] = 1;
vec[2] = 2;
vec[3] = 3;
vec[4] = 4;

// count the odd elements in vec
int result = thrust::count_if(thrust::device, vec.begin(), vec.end(), is_odd());
// result == 2
```

**Template Parameters**:
* **`DerivedPolicy`** The name of the derived execution policy. 
* **`InputIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> must be convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`Predicate`** must be a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`exec`** The execution policy to use for parallelization. 
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`pred`** The predicate. 

**Returns**:
The number of elements where <code>pred</code> is <code>true</code>.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/count">https://en.cppreference.com/w/cpp/algorithm/count</a>

<h3 id="function-count-if">
Function <code>thrust::count&#95;if</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Predicate&gt;</span>
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">thrust::iterator_traits</a>< InputIterator >::difference_type </span><span><b>count_if</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;Predicate pred);</span></code>
<code>count&#95;if</code> finds the number of elements in <code>[first,last)</code> for which a predicate is <code>true</code>. More precisely, <code>count&#95;if</code> returns the number of iterators <code>i</code> in <code>[first, last)</code> such that <code>pred(&#42;i) == true</code>.


The following code snippet demonstrates how to use <code>count</code> to count the number of odd numbers in a range. 

```cpp
#include <thrust/count.h>
#include <thrust/device_vector.h>
...
struct is_odd
{
  __host__ __device__
  bool operator()(int &x)
  {
    return x & 1;
  }
};
...
// fill a device_vector with even & odd numbers
thrust::device_vector<int> vec(5);
vec[0] = 0;
vec[1] = 1;
vec[2] = 2;
vec[3] = 3;
vec[4] = 4;

// count the odd elements in vec
int result = thrust::count_if(vec.begin(), vec.end(), is_odd());
// result == 2
```

**Template Parameters**:
* **`InputIterator`** must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and <code>InputIterator's</code><code>value&#95;type</code> must be convertible to <code>Predicate's</code><code>argument&#95;type</code>. 
* **`Predicate`** must be a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.

**Function Parameters**:
* **`first`** The beginning of the sequence. 
* **`last`** The end of the sequence. 
* **`pred`** The predicate. 

**Returns**:
The number of elements where <code>pred</code> is <code>true</code>.

**See**:
<a href="https://en.cppreference.com/w/cpp/algorithm/count">https://en.cppreference.com/w/cpp/algorithm/count</a>


