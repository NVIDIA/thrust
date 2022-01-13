---
title: Iterators
parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Iterators

## Groups

* **[Fancy Iterators]({{ site.baseurl }}/api/groups/group__fancyiterator.html)**
* **[Iterator Tags]({{ site.baseurl }}/api/groups/group__iterator__tags.html)**

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Distance&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__iterators.html#function-advance">thrust::advance</a></b>(InputIterator & i,</span>
<span>&nbsp;&nbsp;Distance n);</span>
<br>
<span>template &lt;typename InputIterator&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">thrust::iterator_traits</a>< InputIterator >::difference_type </span><span><b><a href="{{ site.baseurl }}/api/groups/group__iterators.html#function-distance">thrust::distance</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last);</span>
</code>

## Functions

<h3 id="function-advance">
Function <code>thrust::advance</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator,</span>
<span>&nbsp;&nbsp;typename Distance&gt;</span>
<span>__host__ __device__ void </span><span><b>advance</b>(InputIterator & i,</span>
<span>&nbsp;&nbsp;Distance n);</span></code>
<code>advance(i, n)</code> increments the iterator <code>i</code> by the distance <code>n</code>. If <code>n &gt; 0</code> it is equivalent to executing <code>++i</code><code>n</code> times, and if <code>n &lt; 0</code> it is equivalent to executing <code>&ndash;i</code><code>n</code> times. If <code>n == 0</code>, the call has no effect.


The following code snippet demonstrates how to use <code>advance</code> to increment an iterator a given number of times.



```cpp
#include <thrust/advance.h>
#include <thrust/device_vector.h>
...
thrust::device_vector<int> vec(13);
thrust::device_vector<int>::iterator iter = vec.begin();

thrust::advance(iter, 7);

// iter - vec.begin() == 7
```

**Template Parameters**:
* **`InputIterator`** is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. 
* **`Distance`** is an integral type that is convertible to <code>InputIterator's</code> distance type.

**Function Parameters**:
* **`i`** The iterator to be advanced. 
* **`n`** The distance by which to advance the iterator.

**Preconditions**:
<code>n</code> shall be negative only for bidirectional and random access iterators.

**See**:
<a href="https://en.cppreference.com/w/cpp/iterator/advance">https://en.cppreference.com/w/cpp/iterator/advance</a>

<h3 id="function-distance">
Function <code>thrust::distance</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">thrust::iterator_traits</a>< InputIterator >::difference_type </span><span><b>distance</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last);</span></code>
<code>distance</code> finds the distance between <code>first</code> and <code>last</code>, i.e. the number of times that <code>first</code> must be incremented until it is equal to <code>last</code>.


The following code snippet demonstrates how to use <code>distance</code> to compute the distance to one iterator from another.



```cpp
#include <thrust/distance.h>
#include <thrust/device_vector.h>
...
thrust::device_vector<int> vec(13);
thrust::device_vector<int>::iterator iter1 = vec.begin();
thrust::device_vector<int>::iterator iter2 = iter1 + 7;

int d = thrust::distance(iter1, iter2);

// d is 7
```

**Template Parameters**:
**`InputIterator`**: is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.

**Function Parameters**:
* **`first`** The beginning of an input range of interest. 
* **`last`** The end of an input range of interest. 

**Preconditions**:
If <code>InputIterator</code> meets the requirements of random access iterator, <code>last</code> shall be reachable from <code>first</code> or <code>first</code> shall be reachable from <code>last</code>; otherwise, <code>last</code> shall be reachable from <code>first</code>.

**Returns**:
The distance between the beginning and end of the input range.

**See**:
<a href="https://en.cppreference.com/w/cpp/iterator/distance">https://en.cppreference.com/w/cpp/iterator/distance</a>


