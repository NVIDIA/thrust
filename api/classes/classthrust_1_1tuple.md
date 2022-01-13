---
title: thrust::tuple
summary: tuple is a class template that can be instantiated with up to ten arguments. Each template argument specifies the type of element in the tuple. Consequently, tuples are heterogeneous, fixed-size collections of values. An instantiation of tuple with two arguments is similar to an instantiation of pair with the same two arguments. Individual elements of a tuple may be accessed with the get function. 
parent: Tuple
grand_parent: Utility
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::tuple`

<code>tuple</code> is a class template that can be instantiated with up to ten arguments. Each template argument specifies the type of element in the <code>tuple</code>. Consequently, tuples are heterogeneous, fixed-size collections of values. An instantiation of <code>tuple</code> with two arguments is similar to an instantiation of <code>pair</code> with the same two arguments. Individual elements of a <code>tuple</code> may be accessed with the <code>get</code> function. 


The following code snippet demonstrates how to create a new <code>tuple</code> object and inspect and modify the value of its elements.



```cpp
#include <thrust/tuple.h>
#include <iostream>

int main() {
  // Create a tuple containing an `int`, a `float`, and a string.
  thrust::tuple<int, float, const char*> t(13, 0.1f, "thrust");

  // Individual members are accessed with the free function `get`.
  std::cout << "The first element's value is " << thrust::get<0>(t) << std::endl;

  // ... or the member function `get`.
  std::cout << "The second element's value is " << t.get<1>() << std::endl;

  // We can also modify elements with the same function.
  thrust::get<0>(t) += 10;
}
```

**Template Parameters**:
**`TN`**: The type of the <code>N</code><code>tuple</code> element. Thrust's <code>tuple</code> type currently supports up to ten elements.

**See**:
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">pair</a>
* get 
* make_tuple 
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1tuple__element.html">tuple_element</a>
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1tuple__size.html">tuple_size</a>
* tie 

<code class="doxybook">
<span>#include <thrust/tuple.h></span><br>
<span>template &lt;class T0,</span>
<span>&nbsp;&nbsp;class T1,</span>
<span>&nbsp;&nbsp;class T2,</span>
<span>&nbsp;&nbsp;class T3,</span>
<span>&nbsp;&nbsp;class T4,</span>
<span>&nbsp;&nbsp;class T5,</span>
<span>&nbsp;&nbsp;class T6,</span>
<span>&nbsp;&nbsp;class T7,</span>
<span>&nbsp;&nbsp;class T8,</span>
<span>&nbsp;&nbsp;class T9&gt;</span>
<span>class thrust::tuple {</span>
<span>public:</span><span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1tuple.html#function-tuple">tuple</a></b>(void);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1tuple.html#function-tuple">tuple</a></b>(typename access_traits< T0 >::parameter_type t0);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1tuple.html#function-tuple">tuple</a></b>(typename access_traits< T0 >::parameter_type t0,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;typename access_traits< T1 >::parameter_type t1);</span>
<br>
<span>&nbsp;&nbsp;template &lt;class U1,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;class U2&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1tuple.html">tuple</a> & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1tuple.html#function-operator=">operator=</a></b>(const <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< U1, U2 > & k);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1tuple.html#function-swap">swap</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1tuple.html">tuple</a> & t);</span>
<span>};</span>
</code>

## Member Functions

<h3 id="function-tuple">
Function <code>thrust::tuple::tuple</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>tuple</b>(void);</span></code>
<code>tuple's</code> no-argument constructor initializes each element. 

<h3 id="function-tuple">
Function <code>thrust::tuple::tuple</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>tuple</b>(typename access_traits< T0 >::parameter_type t0);</span></code>
<code>tuple's</code> one-argument constructor copy constructs the first element from the given parameter and intializes all other elements. 

**Function Parameters**:
**`t0`**: The value to assign to this <code>tuple's</code> first element. 

<h3 id="function-tuple">
Function <code>thrust::tuple::tuple</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>tuple</b>(typename access_traits< T0 >::parameter_type t0,</span>
<span>&nbsp;&nbsp;typename access_traits< T1 >::parameter_type t1);</span></code>
<code>tuple's</code> one-argument constructor copy constructs the first two elements from the given parameters and intializes all other elements. 

**Note**:
<code>tuple's</code> constructor has ten variants of this form, the rest of which are ommitted here for brevity. 

**Function Parameters**:
* **`t0`** The value to assign to this <code>tuple's</code> first element. 
* **`t1`** The value to assign to this <code>tuple's</code> second element. 

<h3 id="function-operator=">
Function <code>thrust::tuple::operator=</code>
</h3>

<code class="doxybook">
<span>template &lt;class U1,</span>
<span>&nbsp;&nbsp;class U2&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1tuple.html">tuple</a> & </span><span><b>operator=</b>(const <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a>< U1, U2 > & k);</span></code>
This assignment operator allows assigning the first two elements of this <code>tuple</code> from a <code>pair</code>. 

**Function Parameters**:
**`k`**: A <code>pair</code> to assign from. 

<h3 id="function-swap">
Function <code>thrust::tuple::swap</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ void </span><span><b>swap</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1tuple.html">tuple</a> & t);</span></code>
<code>swap</code> swaps the elements of two <code>tuple</code>s.

**Function Parameters**:
**`t`**: The other <code>tuple</code> with which to swap. 


