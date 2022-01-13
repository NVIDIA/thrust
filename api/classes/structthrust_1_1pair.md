---
title: thrust::pair
parent: Pair
grand_parent: Utility
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `thrust::pair`

<code>pair</code> is a generic data structure encapsulating a heterogeneous pair of values.

**Template Parameters**:
* **`T1`** The type of <code>pair's</code> first object type. There are no requirements on the type of <code>T1</code>. <code>T1</code>'s type is provided by <code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html#typedef-first-type">pair::first&#95;type</a></code>.
* **`T2`** The type of <code>pair's</code> second object type. There are no requirements on the type of <code>T2</code>. <code>T2</code>'s type is provided by <code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html#typedef-second-type">pair::second&#95;type</a></code>. 

<code class="doxybook">
<span>#include <thrust/pair.h></span><br>
<span>template &lt;typename T1,</span>
<span>&nbsp;&nbsp;typename T2&gt;</span>
<span>struct thrust::pair {</span>
<span>public:</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html#typedef-first-type">first&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html#typedef-second-type">second&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html#typedef-first-type">first_type</a> <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html#variable-first">first</a></b>;</span>
<br>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html#typedef-second-type">second_type</a> <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html#variable-second">second</a></b>;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html#function-pair">pair</a></b>(void);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html#function-pair">pair</a></b>(const T1 & x,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const T2 & y);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U1,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typename U2&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html#function-pair">pair</a></b>(const <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">pair</a>< U1, U2 > & p);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U1,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typename U2&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html#function-pair">pair</a></b>(const std::pair< U1, U2 > & p);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html#function-swap">swap</a></b>(<a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">pair</a> & p);</span>
<span>};</span>
</code>

## Member Types

<h3 id="typedef-first-type">
Typedef <code>thrust::pair::first&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef T1<b>first_type</b>;</span></code>
<code>first&#95;type</code> is the type of <code>pair's</code> first object type. 

<h3 id="typedef-second-type">
Typedef <code>thrust::pair::second&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef T2<b>second_type</b>;</span></code>
<code>second&#95;type</code> is the type of <code>pair's</code> second object type. 


## Member Variables

<h3 id="variable-first">
Variable <code>thrust::pair::first</code>
</h3>

<code class="doxybook">
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html#typedef-first-type">first_type</a> <b>first</b>;</span></code>
The <code>pair's</code> first object. 

<h3 id="variable-second">
Variable <code>thrust::pair::second</code>
</h3>

<code class="doxybook">
<span><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html#typedef-second-type">second_type</a> <b>second</b>;</span></code>
The <code>pair's</code> second object. 


## Member Functions

<h3 id="function-pair">
Function <code>thrust::pair::pair</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>pair</b>(void);</span></code>
<code>pair's</code> default constructor constructs <code>first</code> and <code>second</code> using <code>first&#95;type</code> & <code>second&#95;type's</code> default constructors, respectively. 

<h3 id="function-pair">
Function <code>thrust::pair::pair</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>pair</b>(const T1 & x,</span>
<span>&nbsp;&nbsp;const T2 & y);</span></code>
This constructor accepts two objects to copy into this <code>pair</code>.

**Function Parameters**:
* **`x`** The object to copy into <code>first</code>. 
* **`y`** The object to copy into <code>second</code>. 

<h3 id="function-pair">
Function <code>thrust::pair::pair</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U1,</span>
<span>&nbsp;&nbsp;typename U2&gt;</span>
<span>__host__ __device__ </span><span><b>pair</b>(const <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">pair</a>< U1, U2 > & p);</span></code>
This copy constructor copies from a <code>pair</code> whose types are convertible to this <code>pair's</code><code>first&#95;type</code> and <code>second&#95;type</code>, respectively.

**Template Parameters**:
* **`U1`** is convertible to <code>first&#95;type</code>. 
* **`U2`** is convertible to <code>second&#95;type</code>. 

**Function Parameters**:
**`p`**: The <code>pair</code> to copy from.

<h3 id="function-pair">
Function <code>thrust::pair::pair</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U1,</span>
<span>&nbsp;&nbsp;typename U2&gt;</span>
<span>__host__ __device__ </span><span><b>pair</b>(const std::pair< U1, U2 > & p);</span></code>
This copy constructor copies from a <code>std::pair</code> whose types are convertible to this <code>pair's</code><code>first&#95;type</code> and <code>second&#95;type</code>, respectively.

**Template Parameters**:
* **`U1`** is convertible to <code>first&#95;type</code>. 
* **`U2`** is convertible to <code>second&#95;type</code>. 

**Function Parameters**:
**`p`**: The <code>std::pair</code> to copy from.

<h3 id="function-swap">
Function <code>thrust::pair::swap</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ void </span><span><b>swap</b>(<a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">pair</a> & p);</span></code>
<code>swap</code> swaps the elements of two <code>pair</code>s.

**Function Parameters**:
**`p`**: The other <code>pair</code> with which to swap. 


