---
title: thrust::permutation_iterator
parent: Fancy Iterators
grand_parent: Iterators
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::permutation_iterator`

<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1permutation__iterator.html">permutation&#95;iterator</a></code> is an iterator which represents a pointer into a reordered view of a given range. <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1permutation__iterator.html">permutation&#95;iterator</a></code> is an imprecise name; the reordered view need not be a strict permutation. This iterator is useful for fusing a scatter or gather operation with other algorithms.

This iterator takes two arguments:



* an iterator to the range <code>V</code> on which the "permutation" will be applied
* the reindexing scheme that defines how the elements of <code>V</code> will be permuted.
Note that <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1permutation__iterator.html">permutation&#95;iterator</a></code> is not limited to strict permutations of the given range <code>V</code>. The distance between begin and end of the reindexing iterators is allowed to be smaller compared to the size of the range <code>V</code>, in which case the <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1permutation__iterator.html">permutation&#95;iterator</a></code> only provides a "permutation" of a subrange of <code>V</code>. The indices neither need to be unique. In this same context, it must be noted that the past-the-end <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1permutation__iterator.html">permutation&#95;iterator</a></code> is completely defined by means of the past-the-end iterator to the indices.

The following code snippet demonstrates how to create a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1permutation__iterator.html">permutation&#95;iterator</a></code> which represents a reordering of the contents of a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device&#95;vector</a></code>.



```cpp
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/device_vector.h>
...
thrust::device_vector<float> values(8);
values[0] = 10.0f;
values[1] = 20.0f;
values[2] = 30.0f;
values[3] = 40.0f;
values[4] = 50.0f;
values[5] = 60.0f;
values[6] = 70.0f;
values[7] = 80.0f;

thrust::device_vector<int> indices(4);
indices[0] = 2;
indices[1] = 6;
indices[2] = 1;
indices[3] = 3;

typedef thrust::device_vector<float>::iterator ElementIterator;
typedef thrust::device_vector<int>::iterator   IndexIterator;

thrust::permutation_iterator<ElementIterator,IndexIterator> iter(values.begin(), indices.begin());

*iter;   // returns 30.0f;
iter[0]; // returns 30.0f;
iter[1]; // returns 70.0f;
iter[2]; // returns 20.0f;
iter[3]; // returns 40.0f;

// iter[4] is an out-of-bounds error

*iter   = -1.0f; // sets values[2] to -1.0f;
iter[0] = -1.0f; // sets values[2] to -1.0f;
iter[1] = -1.0f; // sets values[6] to -1.0f;
iter[2] = -1.0f; // sets values[1] to -1.0f;
iter[3] = -1.0f; // sets values[3] to -1.0f;

// values is now {10, -1, -1, -1, 50, 60, -1, 80}
```

**Inherits From**:
`thrust::detail::permutation_iterator_base::type`

**See**:
make_permutation_iterator 

<code class="doxybook">
<span>#include <thrust/iterator/permutation_iterator.h></span><br>
<span>template &lt;typename ElementIterator,</span>
<span>&nbsp;&nbsp;typename IndexIterator&gt;</span>
<span>class thrust::permutation&#95;iterator {</span>
<span>public:</span><span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1permutation__iterator.html#function-permutation-iterator">permutation&#95;iterator</a></b>();</span>
<br>
<span>&nbsp;&nbsp;explicit __host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1permutation__iterator.html#function-permutation-iterator">permutation&#95;iterator</a></b>(ElementIterator x,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;IndexIterator y);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename OtherElementIterator,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typename OtherIndexIterator&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1permutation__iterator.html#function-permutation-iterator">permutation&#95;iterator</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1permutation__iterator.html">permutation_iterator</a>< OtherElementIterator, OtherIndexIterator > const & r,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;typename detail::enable_if_convertible< OtherElementIterator, ElementIterator >::type * = 0,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;typename detail::enable_if_convertible< OtherIndexIterator, IndexIterator >::type * = 0);</span>
<span>};</span>
</code>

## Member Functions

<h3 id="function-permutation-iterator">
Function <code>thrust::permutation&#95;iterator::permutation&#95;iterator</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>permutation_iterator</b>();</span></code>
Null constructor calls the null constructor of this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1permutation__iterator.html">permutation&#95;iterator</a>'s</code> element iterator. 

<h3 id="function-permutation-iterator">
Function <code>thrust::permutation&#95;iterator::permutation&#95;iterator</code>
</h3>

<code class="doxybook">
<span>explicit __host__ __device__ </span><span><b>permutation_iterator</b>(ElementIterator x,</span>
<span>&nbsp;&nbsp;IndexIterator y);</span></code>
Constructor accepts an <code>ElementIterator</code> into a range of values and an <code>IndexIterator</code> into a range of indices defining the indexing scheme on the values.

**Function Parameters**:
* **`x`** An <code>ElementIterator</code> pointing this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1permutation__iterator.html">permutation&#95;iterator</a>'s</code> range of values. 
* **`y`** An <code>IndexIterator</code> pointing to an indexing scheme to use on <code>x</code>. 

<h3 id="function-permutation-iterator">
Function <code>thrust::permutation&#95;iterator::permutation&#95;iterator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename OtherElementIterator,</span>
<span>&nbsp;&nbsp;typename OtherIndexIterator&gt;</span>
<span>__host__ __device__ </span><span><b>permutation_iterator</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1permutation__iterator.html">permutation_iterator</a>< OtherElementIterator, OtherIndexIterator > const & r,</span>
<span>&nbsp;&nbsp;typename detail::enable_if_convertible< OtherElementIterator, ElementIterator >::type * = 0,</span>
<span>&nbsp;&nbsp;typename detail::enable_if_convertible< OtherIndexIterator, IndexIterator >::type * = 0);</span></code>
Copy constructor accepts a related <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1permutation__iterator.html">permutation&#95;iterator</a></code>. 

**Function Parameters**:
**`r`**: A compatible <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1permutation__iterator.html">permutation&#95;iterator</a></code> to copy from. 


