---
title: thrust::zip_iterator
parent: Fancy Iterators
grand_parent: Iterators
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::zip_iterator`

<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__iterator.html">zip&#95;iterator</a></code> is an iterator which represents a pointer into a range of <code>tuples</code> whose elements are themselves taken from a <code>tuple</code> of input iterators. This iterator is useful for creating a virtual array of structures while achieving the same performance and bandwidth as the structure of arrays idiom. <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__iterator.html">zip&#95;iterator</a></code> also facilitates kernel fusion by providing a convenient means of amortizing the execution of the same operation over multiple ranges.

The following code snippet demonstrates how to create a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__iterator.html">zip&#95;iterator</a></code> which represents the result of "zipping" multiple ranges together.



```cpp
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
...
thrust::device_vector<int> int_v(3);
int_v[0] = 0; int_v[1] = 1; int_v[2] = 2;

thrust::device_vector<float> float_v(3);
float_v[0] = 0.0f; float_v[1] = 1.0f; float_v[2] = 2.0f;

thrust::device_vector<char> char_v(3);
char_v[0] = 'a'; char_v[1] = 'b'; char_v[2] = 'c';

// typedef these iterators for shorthand
typedef thrust::device_vector<int>::iterator   IntIterator;
typedef thrust::device_vector<float>::iterator FloatIterator;
typedef thrust::device_vector<char>::iterator  CharIterator;

// typedef a tuple of these iterators
typedef thrust::tuple<IntIterator, FloatIterator, CharIterator> IteratorTuple;

// typedef the zip_iterator of this tuple
typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

// finally, create the zip_iterator
ZipIterator iter(thrust::make_tuple(int_v.begin(), float_v.begin(), char_v.begin()));

*iter;   // returns (0, 0.0f, 'a')
iter[0]; // returns (0, 0.0f, 'a')
iter[1]; // returns (1, 1.0f, 'b')
iter[2]; // returns (2, 2.0f, 'c')

thrust::get<0>(iter[2]); // returns 2
thrust::get<1>(iter[0]); // returns 0.0f
thrust::get<2>(iter[1]); // returns 'b'

// iter[3] is an out-of-bounds error
```

Defining the type of a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__iterator.html">zip&#95;iterator</a></code> can be complex. The next code example demonstrates how to use the <code>make&#95;zip&#95;iterator</code> function with the <code>make&#95;tuple</code> function to avoid explicitly specifying the type of the <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__iterator.html">zip&#95;iterator</a></code>. This example shows how to use <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__iterator.html">zip&#95;iterator</a></code> to copy multiple ranges with a single call to <code>thrust::copy</code>.



```cpp
#include <thrust/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>

int main()
{
  thrust::device_vector<int> int_in(3), int_out(3);
  int_in[0] = 0;
  int_in[1] = 1;
  int_in[2] = 2;

  thrust::device_vector<float> float_in(3), float_out(3);
  float_in[0] =  0.0f;
  float_in[1] = 10.0f;
  float_in[2] = 20.0f;

  thrust::copy(thrust::make_zip_iterator(thrust::make_tuple(int_in.begin(), float_in.begin())),
               thrust::make_zip_iterator(thrust::make_tuple(int_in.end(),   float_in.end())),
               thrust::make_zip_iterator(thrust::make_tuple(int_out.begin(),float_out.begin())));

  // int_out is now [0, 1, 2]
  // float_out is now [0.0f, 10.0f, 20.0f]

  return 0;
}
```

**Inherits From**:
`detail::zip_iterator_base::type`

**See**:
* make_zip_iterator 
* make_tuple 
* <a href="{{ site.baseurl }}/api/classes/classthrust_1_1tuple.html">tuple</a>
* get 

<code class="doxybook">
<span>#include <thrust/iterator/zip_iterator.h></span><br>
<span>template &lt;typename IteratorTuple&gt;</span>
<span>class thrust::zip&#95;iterator {</span>
<span>public:</span><span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__iterator.html#function-zip-iterator">zip&#95;iterator</a></b>();</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__iterator.html#function-zip-iterator">zip&#95;iterator</a></b>(IteratorTuple iterator_tuple);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename OtherIteratorTuple&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__iterator.html#function-zip-iterator">zip&#95;iterator</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__iterator.html">zip_iterator</a>< OtherIteratorTuple > & other,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;typename thrust::detail::enable_if_convertible< OtherIteratorTuple, IteratorTuple >::type * = 0);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ const IteratorTuple & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__iterator.html#function-get-iterator-tuple">get&#95;iterator&#95;tuple</a></b>() const;</span>
<span>};</span>
</code>

## Member Functions

<h3 id="function-zip-iterator">
Function <code>thrust::zip&#95;iterator::zip&#95;iterator</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>zip_iterator</b>();</span></code>
Null constructor does nothing. 

<h3 id="function-zip-iterator">
Function <code>thrust::zip&#95;iterator::zip&#95;iterator</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>zip_iterator</b>(IteratorTuple iterator_tuple);</span></code>
This constructor creates a new <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__iterator.html">zip&#95;iterator</a></code> from a <code>tuple</code> of iterators.

**Function Parameters**:
**`iterator_tuple`**: The <code>tuple</code> of iterators to copy from. 

<h3 id="function-zip-iterator">
Function <code>thrust::zip&#95;iterator::zip&#95;iterator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename OtherIteratorTuple&gt;</span>
<span>__host__ __device__ </span><span><b>zip_iterator</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__iterator.html">zip_iterator</a>< OtherIteratorTuple > & other,</span>
<span>&nbsp;&nbsp;typename thrust::detail::enable_if_convertible< OtherIteratorTuple, IteratorTuple >::type * = 0);</span></code>
This copy constructor creates a new <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__iterator.html">zip&#95;iterator</a></code> from another <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__iterator.html">zip&#95;iterator</a></code>.

**Function Parameters**:
**`other`**: The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__iterator.html">zip&#95;iterator</a></code> to copy. 

<h3 id="function-get-iterator-tuple">
Function <code>thrust::zip&#95;iterator::get&#95;iterator&#95;tuple</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ const IteratorTuple & </span><span><b>get_iterator_tuple</b>() const;</span></code>
This method returns a <code>const</code> reference to this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__iterator.html">zip&#95;iterator</a>'s</code><code>tuple</code> of iterators.

**Returns**:
A <code>const</code> reference to this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1zip__iterator.html">zip&#95;iterator</a>'s</code><code>tuple</code> of iterators. 


