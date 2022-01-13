---
title: Tuple
parent: Utility
grand_parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Tuple

<code class="doxybook">
<span>template &lt;size_t N,</span>
<span>&nbsp;&nbsp;class T&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1tuple__element.html">thrust::tuple&#95;element</a></b>;</span>
<br>
<span>template &lt;typename Pair&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1tuple__size.html">thrust::tuple&#95;size</a></b>;</span>
<br>
<span class="doxybook-comment">/* <code>tuple</code> is a class template that can be instantiated with up to ten arguments. Each template argument specifies the type of element in the <code>tuple</code>. Consequently, tuples are heterogeneous, fixed-size collections of values. An instantiation of <code>tuple</code> with two arguments is similar to an instantiation of <code>pair</code> with the same two arguments. Individual elements of a <code>tuple</code> may be accessed with the <code>get</code> function.  */</span><span>template &lt;class T0,</span>
<span>&nbsp;&nbsp;class T1,</span>
<span>&nbsp;&nbsp;class T2,</span>
<span>&nbsp;&nbsp;class T3,</span>
<span>&nbsp;&nbsp;class T4,</span>
<span>&nbsp;&nbsp;class T5,</span>
<span>&nbsp;&nbsp;class T6,</span>
<span>&nbsp;&nbsp;class T7,</span>
<span>&nbsp;&nbsp;class T8,</span>
<span>&nbsp;&nbsp;class T9&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1tuple.html">thrust::tuple</a></b>;</span>
<br>
<span>template &lt;int N,</span>
<span>&nbsp;&nbsp;class HT,</span>
<span>&nbsp;&nbsp;class TT&gt;</span>
<span>__host__ __device__ access_traits< typenametuple_element< N, detail::cons< HT, TT > >::type >::non_const_type </span><span><b><a href="{{ site.baseurl }}/api/groups/group__tuple.html#function-get">thrust::get</a></b>(detail::cons< HT, TT > & t);</span>
<br>
<span>template &lt;int N,</span>
<span>&nbsp;&nbsp;class HT,</span>
<span>&nbsp;&nbsp;class TT&gt;</span>
<span>__host__ __device__ access_traits< typenametuple_element< N, detail::cons< HT, TT > >::type >::const_type </span><span><b><a href="{{ site.baseurl }}/api/groups/group__tuple.html#function-get">thrust::get</a></b>(const detail::cons< HT, TT > & t);</span>
<br>
<span>template &lt;class T0&gt;</span>
<span>__host__ __device__ detail::make_tuple_mapper< T0 >::type </span><span><b><a href="{{ site.baseurl }}/api/groups/group__tuple.html#function-make-tuple">thrust::make&#95;tuple</a></b>(const T0 & t0);</span>
<br>
<span>template &lt;class T0,</span>
<span>&nbsp;&nbsp;class T1&gt;</span>
<span>__host__ __device__ detail::make_tuple_mapper< T0, T1 >::type </span><span><b><a href="{{ site.baseurl }}/api/groups/group__tuple.html#function-make-tuple">thrust::make&#95;tuple</a></b>(const T0 & t0,</span>
<span>&nbsp;&nbsp;const T1 & t1);</span>
<br>
<span>template &lt;typename T0&gt;</span>
<span>__host__ __device__ tuple< T0 & > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__tuple.html#function-tie">thrust::tie</a></b>(T0 & t0);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ tuple< T0 &, T1 & > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__tuple.html#function-tie">thrust::tie</a></b>(T0 & t0,</span>
<span>&nbsp;&nbsp;T1 & t1);</span>
<br>
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1,</span>
<span>&nbsp;&nbsp;typename T2,</span>
<span>&nbsp;&nbsp;typename T3,</span>
<span>&nbsp;&nbsp;typename T4,</span>
<span>&nbsp;&nbsp;typename T5,</span>
<span>&nbsp;&nbsp;typename T6,</span>
<span>&nbsp;&nbsp;typename T7,</span>
<span>&nbsp;&nbsp;typename T8,</span>
<span>&nbsp;&nbsp;typename T9,</span>
<span>&nbsp;&nbsp;typename U0,</span>
<span>&nbsp;&nbsp;typename U1,</span>
<span>&nbsp;&nbsp;typename U2,</span>
<span>&nbsp;&nbsp;typename U3,</span>
<span>&nbsp;&nbsp;typename U4,</span>
<span>&nbsp;&nbsp;typename U5,</span>
<span>&nbsp;&nbsp;typename U6,</span>
<span>&nbsp;&nbsp;typename U7,</span>
<span>&nbsp;&nbsp;typename U8,</span>
<span>&nbsp;&nbsp;typename U9&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__tuple.html#function-swap">thrust::swap</a></b>(tuple< T0, T1, T2, T3, T4, T5, T6, T7, T8, T9 > & x,</span>
<span>&nbsp;&nbsp;tuple< U0, U1, U2, U3, U4, U5, U6, U7, U8, U9 > & y);</span>
</code>

## Member Classes

<h3 id="struct-thrusttuple-element">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1tuple__element.html">Struct <code>thrust::tuple&#95;element</code>
</a>
</h3>

<h3 id="struct-thrusttuple-size">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1tuple__size.html">Struct <code>thrust::tuple&#95;size</code>
</a>
</h3>

<h3 id="class-thrusttuple">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1tuple.html">Class <code>thrust::tuple</code>
</a>
</h3>

<code>tuple</code> is a class template that can be instantiated with up to ten arguments. Each template argument specifies the type of element in the <code>tuple</code>. Consequently, tuples are heterogeneous, fixed-size collections of values. An instantiation of <code>tuple</code> with two arguments is similar to an instantiation of <code>pair</code> with the same two arguments. Individual elements of a <code>tuple</code> may be accessed with the <code>get</code> function. 


## Functions

<h3 id="function-get">
Function <code>thrust::get</code>
</h3>

<code class="doxybook">
<span>template &lt;int N,</span>
<span>&nbsp;&nbsp;class HT,</span>
<span>&nbsp;&nbsp;class TT&gt;</span>
<span>__host__ __device__ access_traits< typenametuple_element< N, detail::cons< HT, TT > >::type >::non_const_type </span><span><b>get</b>(detail::cons< HT, TT > & t);</span></code>
The <code>get</code> function returns a reference to a <code>tuple</code> element of interest.


The following code snippet demonstrates how to use <code>get</code> to print the value of a <code>tuple</code> element.



```cpp
#include <thrust/tuple.h>
#include <iostream>
...
thrust::tuple<int, const char *> t(13, "thrust");

std::cout << "The 1st value of t is " << thrust::get<0>(t) << std::endl;
```

**Template Parameters**:
**`N`**: The index of the element of interest.

**Function Parameters**:
**`t`**: A reference to a <code>tuple</code> of interest. 

**Returns**:
A reference to <code>t's</code><code>N</code>th element.

**See**:
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">pair</a>
* <a href="{{ site.baseurl }}/api/classes/classthrust_1_1tuple.html">tuple</a>

<h3 id="function-get">
Function <code>thrust::get</code>
</h3>

<code class="doxybook">
<span>template &lt;int N,</span>
<span>&nbsp;&nbsp;class HT,</span>
<span>&nbsp;&nbsp;class TT&gt;</span>
<span>__host__ __device__ access_traits< typenametuple_element< N, detail::cons< HT, TT > >::type >::const_type </span><span><b>get</b>(const detail::cons< HT, TT > & t);</span></code>
The <code>get</code> function returns a <code>const</code> reference to a <code>tuple</code> element of interest.


The following code snippet demonstrates how to use <code>get</code> to print the value of a <code>tuple</code> element.



```cpp
#include <thrust/tuple.h>
#include <iostream>
...
thrust::tuple<int, const char *> t(13, "thrust");

std::cout << "The 1st value of t is " << thrust::get<0>(t) << std::endl;
```

**Template Parameters**:
**`N`**: The index of the element of interest.

**Function Parameters**:
**`t`**: A reference to a <code>tuple</code> of interest. 

**Returns**:
A <code>const</code> reference to <code>t's</code><code>N</code>th element.

**See**:
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">pair</a>
* <a href="{{ site.baseurl }}/api/classes/classthrust_1_1tuple.html">tuple</a>

<h3 id="function-make-tuple">
Function <code>thrust::make&#95;tuple</code>
</h3>

<code class="doxybook">
<span>template &lt;class T0&gt;</span>
<span>__host__ __device__ detail::make_tuple_mapper< T0 >::type </span><span><b>make_tuple</b>(const T0 & t0);</span></code>
This version of <code>make&#95;tuple</code> creates a new <code>tuple</code> object from a single object.

**Function Parameters**:
**`t0`**: The object to copy from. 

**Returns**:
A <code>tuple</code> object with a single member which is a copy of <code>t0</code>. 

<h3 id="function-make-tuple">
Function <code>thrust::make&#95;tuple</code>
</h3>

<code class="doxybook">
<span>template &lt;class T0,</span>
<span>&nbsp;&nbsp;class T1&gt;</span>
<span>__host__ __device__ detail::make_tuple_mapper< T0, T1 >::type </span><span><b>make_tuple</b>(const T0 & t0,</span>
<span>&nbsp;&nbsp;const T1 & t1);</span></code>
This version of <code>make&#95;tuple</code> creates a new <code>tuple</code> object from two objects.

**Note**:
<code>make&#95;tuple</code> has ten variants, the rest of which are omitted here for brevity. 

**Function Parameters**:
* **`t0`** The first object to copy from. 
* **`t1`** The second object to copy from. 

**Returns**:
A <code>tuple</code> object with two members which are copies of <code>t0</code> and <code>t1</code>.

<h3 id="function-tie">
Function <code>thrust::tie</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0&gt;</span>
<span>__host__ __device__ tuple< T0 & > </span><span><b>tie</b>(T0 & t0);</span></code>
This version of <code>tie</code> creates a new <code>tuple</code> whose single element is a reference which refers to this function's argument.

**Function Parameters**:
**`t0`**: The object to reference. 

**Returns**:
A <code>tuple</code> object with one member which is a reference to <code>t0</code>. 

<h3 id="function-tie">
Function <code>thrust::tie</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1&gt;</span>
<span>__host__ __device__ tuple< T0 &, T1 & > </span><span><b>tie</b>(T0 & t0,</span>
<span>&nbsp;&nbsp;T1 & t1);</span></code>
This version of <code>tie</code> creates a new <code>tuple</code> of references object which refers to this function's arguments.

**Note**:
<code>tie</code> has ten variants, the rest of which are omitted here for brevity. 

**Function Parameters**:
* **`t0`** The first object to reference. 
* **`t1`** The second object to reference. 

**Returns**:
A <code>tuple</code> object with two members which are references to <code>t0</code> and <code>t1</code>.

<h3 id="function-swap">
Function <code>thrust::swap</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T0,</span>
<span>&nbsp;&nbsp;typename T1,</span>
<span>&nbsp;&nbsp;typename T2,</span>
<span>&nbsp;&nbsp;typename T3,</span>
<span>&nbsp;&nbsp;typename T4,</span>
<span>&nbsp;&nbsp;typename T5,</span>
<span>&nbsp;&nbsp;typename T6,</span>
<span>&nbsp;&nbsp;typename T7,</span>
<span>&nbsp;&nbsp;typename T8,</span>
<span>&nbsp;&nbsp;typename T9,</span>
<span>&nbsp;&nbsp;typename U0,</span>
<span>&nbsp;&nbsp;typename U1,</span>
<span>&nbsp;&nbsp;typename U2,</span>
<span>&nbsp;&nbsp;typename U3,</span>
<span>&nbsp;&nbsp;typename U4,</span>
<span>&nbsp;&nbsp;typename U5,</span>
<span>&nbsp;&nbsp;typename U6,</span>
<span>&nbsp;&nbsp;typename U7,</span>
<span>&nbsp;&nbsp;typename U8,</span>
<span>&nbsp;&nbsp;typename U9&gt;</span>
<span>__host__ __device__ void </span><span><b>swap</b>(tuple< T0, T1, T2, T3, T4, T5, T6, T7, T8, T9 > & x,</span>
<span>&nbsp;&nbsp;tuple< U0, U1, U2, U3, U4, U5, U6, U7, U8, U9 > & y);</span></code>
<code>swap</code> swaps the contents of two <code>tuple</code>s.

**Function Parameters**:
* **`x`** The first <code>tuple</code> to swap. 
* **`y`** The second <code>tuple</code> to swap. 


