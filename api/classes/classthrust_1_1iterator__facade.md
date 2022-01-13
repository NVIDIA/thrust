---
title: thrust::iterator_facade
parent: Fancy Iterators
grand_parent: Iterators
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::iterator_facade`

<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a></code> is a template which allows the programmer to define a novel iterator with a standards-conforming interface which Thrust can use to reason about algorithm acceleration opportunities.

Because most of a standard iterator's interface is defined in terms of a small set of core primitives, <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a></code> defines the non-primitive portion mechanically. In principle a novel iterator could explicitly provide the entire interface in an ad hoc fashion but doing so might be tedious and prone to subtle errors.

Often <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a></code> is too primitive a tool to use for defining novel iterators. In these cases, <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__adaptor.html">iterator&#95;adaptor</a></code> or a specific fancy iterator should be used instead.

<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a>'s</code> functionality is derived from and generally equivalent to <code>boost::iterator&#95;facade</code>. The exception is Thrust's addition of the template parameter <code>System</code>, which is necessary to allow Thrust to dispatch an algorithm to one of several parallel backend systems. An additional exception is Thrust's omission of the <code>operator-&gt;</code> member function.

Interested users may refer to <code>boost::iterator&#95;facade</code>'s documentation for usage examples.

**Note**:
<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a>'s</code> arithmetic operator free functions exist with the usual meanings but are omitted here for brevity. 

<code class="doxybook">
<span>#include <thrust/iterator/iterator_facade.h></span><br>
<span>template &lt;typename Derived,</span>
<span>&nbsp;&nbsp;typename Value,</span>
<span>&nbsp;&nbsp;typename System,</span>
<span>&nbsp;&nbsp;typename Traversal,</span>
<span>&nbsp;&nbsp;typename Reference,</span>
<span>&nbsp;&nbsp;typename Difference = std::ptrdiff&#95;t&gt;</span>
<span>class thrust::iterator&#95;facade {</span>
<span>public:</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#typedef-value-type">value&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#typedef-reference">reference</a></b>;</span>
<br>
<span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#typedef-pointer">pointer</a></b>;</span>
<br>
<span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#typedef-difference-type">difference&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#typedef-iterator-category">iterator&#95;category</a></b>;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#typedef-reference">reference</a> </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#function-operator*">operator&#42;</a></b>() const;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#typedef-reference">reference</a> </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#function-operator[]">operator[]</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#typedef-difference-type">difference_type</a> n) const;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ Derived & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#function-operator++">operator++</a></b>();</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ Derived </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#function-operator++">operator++</a></b>(int);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ Derived & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#function-operator--">operator--</a></b>();</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ Derived </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#function-operator--">operator--</a></b>(int);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ Derived & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#function-operator+=">operator+=</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#typedef-difference-type">difference_type</a> n);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ Derived & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#function-operator-=">operator-=</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#typedef-difference-type">difference_type</a> n);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ Derived </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#function-operator-">operator-</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#typedef-difference-type">difference_type</a> n) const;</span>
<span>};</span>
</code>

## Member Types

<h3 id="typedef-value-type">
Typedef <code>thrust::iterator&#95;facade::value&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef thrust::detail::remove_const< Value >::type<b>value_type</b>;</span></code>
The type of element pointed to by <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a></code>. 

<h3 id="typedef-reference">
Typedef <code>thrust::iterator&#95;facade::reference</code>
</h3>

<code class="doxybook">
<span>typedef Reference<b>reference</b>;</span></code>
The return type of <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#function-operator*">iterator&#95;facade::operator&#42;()</a></code>. 

<h3 id="typedef-pointer">
Typedef <code>thrust::iterator&#95;facade::pointer</code>
</h3>

<code class="doxybook">
<span>typedef void<b>pointer</b>;</span></code>
The return type of <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a>'s</code> non-existent <code>operator-&gt;()</code> member function. Unlike <code>boost::iterator&#95;facade</code>, <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a></code> disallows access to the <code>value&#95;type's</code> members through expressions of the form <code>iter-&gt;member</code>. <code>pointer</code> is defined to <code>void</code> to indicate that these expressions are not allowed. This limitation may be relaxed in a future version of Thrust. 

<h3 id="typedef-difference-type">
Typedef <code>thrust::iterator&#95;facade::difference&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef Difference<b>difference_type</b>;</span></code>
The type of expressions of the form <code>x - y</code> where <code>x</code> and <code>y</code> are of type <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a></code>. 

<h3 id="typedef-iterator-category">
Typedef <code>thrust::iterator&#95;facade::iterator&#95;category</code>
</h3>

<code class="doxybook">
<span>typedef thrust::detail::iterator_facade_category< System, Traversal, Value, Reference >::type<b>iterator_category</b>;</span></code>
The type of iterator category of <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a></code>. 


## Member Functions

<h3 id="function-operator*">
Function <code>thrust::iterator&#95;facade::operator&#42;</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#typedef-reference">reference</a> </span><span><b>operator*</b>() const;</span></code>
<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#function-operator*">operator&#42;()</a></code> dereferences this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a></code>. 

**Returns**:
A reference to the element pointed to by this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a></code>. 

<h3 id="function-operator[]">
Function <code>thrust::iterator&#95;facade::operator[]</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#typedef-reference">reference</a> </span><span><b>operator[]</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#typedef-difference-type">difference_type</a> n) const;</span></code>
<code>operator</code>[] performs indexed dereference. 

**Returns**:
A reference to the element <code>n</code> distance away from this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a></code>. 

<h3 id="function-operator++">
Function <code>thrust::iterator&#95;facade::operator++</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ Derived & </span><span><b>operator++</b>();</span></code>
<code>operator++</code> pre-increments this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a></code> to refer to the element in the next position. 

**Returns**:
<code>&#42;this</code>

<h3 id="function-operator++">
Function <code>thrust::iterator&#95;facade::operator++</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ Derived </span><span><b>operator++</b>(int);</span></code>
<code>operator++</code> post-increments this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a></code> and returns a new <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a></code> referring to the element in the next position. 

**Returns**:
A copy of <code>&#42;this</code> before increment. 

<h3 id="function-operator--">
Function <code>thrust::iterator&#95;facade::operator--</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ Derived & </span><span><b>operator--</b>();</span></code>
<code>operator--</code> pre-decrements this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a></code> to refer to the element in the previous position. 

**Returns**:
<code>&#42;this</code>

<h3 id="function-operator--">
Function <code>thrust::iterator&#95;facade::operator--</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ Derived </span><span><b>operator--</b>(int);</span></code>
<code>operator--</code> post-decrements this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a></code> and returns a new <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a></code> referring to the element in the previous position. 

**Returns**:
A copy of <code>&#42;this</code> before decrement. 

<h3 id="function-operator+=">
Function <code>thrust::iterator&#95;facade::operator+=</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ Derived & </span><span><b>operator+=</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#typedef-difference-type">difference_type</a> n);</span></code>
<code>operator+=</code> increments this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a></code> to refer to an element a given distance after its current position. 

**Function Parameters**:
**`n`**: The quantity to increment. 

**Returns**:
<code>&#42;this</code>

<h3 id="function-operator-=">
Function <code>thrust::iterator&#95;facade::operator-=</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ Derived & </span><span><b>operator-=</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#typedef-difference-type">difference_type</a> n);</span></code>
<code>operator-=</code> decrements this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a></code> to refer to an element a given distance before its current postition. 

**Function Parameters**:
**`n`**: The quantity to decrement. 

**Returns**:
<code>&#42;this</code>

<h3 id="function-operator-">
Function <code>thrust::iterator&#95;facade::operator-</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ Derived </span><span><b>operator-</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html#typedef-difference-type">difference_type</a> n) const;</span></code>
<code>operator-</code> subtracts a given quantity from this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a></code> and returns a new <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a></code> referring to the element at the given position before this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a></code>. 

**Function Parameters**:
**`n`**: The quantity to decrement 

**Returns**:
An <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a></code> pointing <code>n</code> elements before this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__facade.html">iterator&#95;facade</a></code>. 


