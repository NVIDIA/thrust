---
title: thrust::reverse_iterator
parent: Fancy Iterators
grand_parent: Iterators
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::reverse_iterator`

<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1reverse__iterator.html">reverse&#95;iterator</a></code> is an iterator which represents a pointer into a reversed view of a given range. In this way, <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1reverse__iterator.html">reverse&#95;iterator</a></code> allows backwards iteration through a bidirectional input range.

It is important to note that although <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1reverse__iterator.html">reverse&#95;iterator</a></code> is constructed from a given iterator, it points to the element preceding it. In this way, the past-the-end <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1reverse__iterator.html">reverse&#95;iterator</a></code> of a given range points to the element preceding the first element of the input range. By the same token, the first <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1reverse__iterator.html">reverse&#95;iterator</a></code> of a given range is constructed from a past-the-end iterator of the original range yet points to the last element of the input.

The following code snippet demonstrates how to create a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1reverse__iterator.html">reverse&#95;iterator</a></code> which represents a reversed view of the contents of a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device&#95;vector</a></code>.



```cpp
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/device_vector.h>
...
thrust::device_vector<float> v(4);
v[0] = 0.0f;
v[1] = 1.0f;
v[2] = 2.0f;
v[3] = 3.0f;

typedef thrust::device_vector<float>::iterator Iterator;

// note that we point the iterator to the *end* of the device_vector
thrust::reverse_iterator<Iterator> iter(values.end());

*iter;   // returns 3.0f;
iter[0]; // returns 3.0f;
iter[1]; // returns 2.0f;
iter[2]; // returns 1.0f;
iter[3]; // returns 0.0f;

// iter[4] is an out-of-bounds error
```

Since reversing a range is a common operation, containers like <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device&#95;vector</a></code> have nested typedefs for declaration shorthand and methods for constructing reverse_iterators. The following code snippet is equivalent to the previous:



```cpp
#include <thrust/device_vector.h>
...
thrust::device_vector<float> v(4);
v[0] = 0.0f;
v[1] = 1.0f;
v[2] = 2.0f;
v[3] = 3.0f;

// we use the nested type reverse_iterator to refer to a reversed view of
// a device_vector and the method rbegin() to create a reverse_iterator pointing
// to the beginning of the reversed device_vector
thrust::device_iterator<float>::reverse_iterator iter = values.rbegin();

*iter;   // returns 3.0f;
iter[0]; // returns 3.0f;
iter[1]; // returns 2.0f;
iter[2]; // returns 1.0f;
iter[3]; // returns 0.0f;

// iter[4] is an out-of-bounds error

// similarly, rend() points to the end of the reversed sequence:
assert(values.rend() == (iter + 4));
```

Finally, the following code snippet demonstrates how to use <a href="{{ site.baseurl }}/api/classes/classthrust_1_1reverse__iterator.html">reverse_iterator</a> to perform a reversed prefix sum operation on the contents of a <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device_vector</a>:



```cpp
#include <thrust/device_vector.h>
#include <thrust/scan.h>
...
thrust::device_vector<int> v(5);
v[0] = 0;
v[1] = 1;
v[2] = 2;
v[3] = 3;
v[4] = 4;

thrust::device_vector<int> result(5);

// exclusive scan v into result in reverse
thrust::exclusive_scan(v.rbegin(), v.rend(), result.begin());

// result is now {0, 4, 7, 9, 10}
```

**Inherits From**:
`detail::reverse_iterator_base::type`

**See**:
make_reverse_iterator 

<code class="doxybook">
<span>#include <thrust/iterator/reverse_iterator.h></span><br>
<span>template &lt;typename BidirectionalIterator&gt;</span>
<span>class thrust::reverse&#95;iterator {</span>
<span>public:</span><span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1reverse__iterator.html#function-reverse-iterator">reverse&#95;iterator</a></b>();</span>
<br>
<span>&nbsp;&nbsp;explicit __host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1reverse__iterator.html#function-reverse-iterator">reverse&#95;iterator</a></b>(BidirectionalIterator x);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename OtherBidirectionalIterator&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1reverse__iterator.html#function-reverse-iterator">reverse&#95;iterator</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1reverse__iterator.html">reverse_iterator</a>< OtherBidirectionalIterator > const & r);</span>
<span>};</span>
</code>

## Member Functions

<h3 id="function-reverse-iterator">
Function <code>thrust::reverse&#95;iterator::reverse&#95;iterator</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>reverse_iterator</b>();</span></code>
Default constructor does nothing. 

<h3 id="function-reverse-iterator">
Function <code>thrust::reverse&#95;iterator::reverse&#95;iterator</code>
</h3>

<code class="doxybook">
<span>explicit __host__ __device__ </span><span><b>reverse_iterator</b>(BidirectionalIterator x);</span></code>
<code>Constructor</code> accepts a <code>BidirectionalIterator</code> pointing to a range for this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1reverse__iterator.html">reverse&#95;iterator</a></code> to reverse.

**Function Parameters**:
**`x`**: A <code>BidirectionalIterator</code> pointing to a range to reverse. 

<h3 id="function-reverse-iterator">
Function <code>thrust::reverse&#95;iterator::reverse&#95;iterator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename OtherBidirectionalIterator&gt;</span>
<span>__host__ __device__ </span><span><b>reverse_iterator</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1reverse__iterator.html">reverse_iterator</a>< OtherBidirectionalIterator > const & r);</span></code>
<code>Copy</code> constructor allows construction from a related compatible <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1reverse__iterator.html">reverse&#95;iterator</a></code>.

**Function Parameters**:
**`r`**: A <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1reverse__iterator.html">reverse&#95;iterator</a></code> to copy from. 


