---
title: thrust::counting_iterator
parent: Fancy Iterators
grand_parent: Iterators
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::counting_iterator`

<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1counting__iterator.html">counting&#95;iterator</a></code> is an iterator which represents a pointer into a range of sequentially changing values. This iterator is useful for creating a range filled with a sequence without explicitly storing it in memory. Using <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1counting__iterator.html">counting&#95;iterator</a></code> saves memory capacity and bandwidth.

The following code snippet demonstrates how to create a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1counting__iterator.html">counting&#95;iterator</a></code> whose <code>value&#95;type</code> is <code>int</code> and which sequentially increments by <code>1</code>.



```cpp
#include <thrust/iterator/counting_iterator.h>
...
// create iterators
thrust::counting_iterator<int> first(10);
thrust::counting_iterator<int> last = first + 3;

first[0]   // returns 10
first[1]   // returns 11
first[100] // returns 110

// sum of [first, last)
thrust::reduce(first, last);   // returns 33 (i.e. 10 + 11 + 12)

// initialize vector to [0,1,2,..]
thrust::counting_iterator<int> iter(0);
thrust::device_vector<int> vec(500);
thrust::copy(iter, iter + vec.size(), vec.begin());
```

This next example demonstrates how to use a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1counting__iterator.html">counting&#95;iterator</a></code> with the <code>thrust::copy&#95;if</code> function to compute the indices of the non-zero elements of a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device&#95;vector</a></code>. In this example, we use the <code>make&#95;counting&#95;iterator</code> function to avoid specifying the type of the <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1counting__iterator.html">counting&#95;iterator</a></code>.



```cpp
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>

int main()
{
 // this example computes indices for all the nonzero values in a sequence

 // sequence of zero and nonzero values
 thrust::device_vector<int> stencil(8);
 stencil[0] = 0;
 stencil[1] = 1;
 stencil[2] = 1;
 stencil[3] = 0;
 stencil[4] = 0;
 stencil[5] = 1;
 stencil[6] = 0;
 stencil[7] = 1;

 // storage for the nonzero indices
 thrust::device_vector<int> indices(8);

 // compute indices of nonzero elements
 typedef thrust::device_vector<int>::iterator IndexIterator;

 // use make_counting_iterator to define the sequence [0, 8)
 IndexIterator indices_end = thrust::copy_if(thrust::make_counting_iterator(0),
                                             thrust::make_counting_iterator(8),
                                             stencil.begin(),
                                             indices.begin(),
                                             thrust::identity<int>());
 // indices now contains [1,2,5,7]

 return 0;
}
```

**Inherits From**:
`detail::counting_iterator_base::type`

**See**:
make_counting_iterator 

<code class="doxybook">
<span>#include <thrust/iterator/counting_iterator.h></span><br>
<span>template &lt;typename Incrementable,</span>
<span>&nbsp;&nbsp;typename System = use&#95;default,</span>
<span>&nbsp;&nbsp;typename Traversal = use&#95;default,</span>
<span>&nbsp;&nbsp;typename Difference = use&#95;default&gt;</span>
<span>class thrust::counting&#95;iterator {</span>
<span>};</span>
</code>

