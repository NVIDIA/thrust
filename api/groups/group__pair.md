---
title: Pair
parent: Utility
grand_parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Pair

<code class="doxybook">
<span>template &lt;typename T1,</span>
<span>&nbsp;&nbsp;typename T2&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">thrust::pair</a></b>;</span>
<br>
<span>template &lt;size_t N,</span>
<span>&nbsp;&nbsp;class T&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1tuple__element.html">thrust::tuple&#95;element</a></b>;</span>
<br>
<span>template &lt;typename Pair&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1tuple__size.html">thrust::tuple&#95;size</a></b>;</span>
<br>
<span>template &lt;typename T1,</span>
<span>&nbsp;&nbsp;typename T2&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__pair.html#function-operator==">thrust::operator==</a></b>(const pair< T1, T2 > & x,</span>
<span>&nbsp;&nbsp;const pair< T1, T2 > & y);</span>
<br>
<span>template &lt;typename T1,</span>
<span>&nbsp;&nbsp;typename T2&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__pair.html#function-operator<">thrust::operator&lt;</a></b>(const pair< T1, T2 > & x,</span>
<span>&nbsp;&nbsp;const pair< T1, T2 > & y);</span>
<br>
<span>template &lt;typename T1,</span>
<span>&nbsp;&nbsp;typename T2&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__pair.html#function-operator!=">thrust::operator!=</a></b>(const pair< T1, T2 > & x,</span>
<span>&nbsp;&nbsp;const pair< T1, T2 > & y);</span>
<br>
<span>template &lt;typename T1,</span>
<span>&nbsp;&nbsp;typename T2&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__pair.html#function-operator>">thrust::operator&gt;</a></b>(const pair< T1, T2 > & x,</span>
<span>&nbsp;&nbsp;const pair< T1, T2 > & y);</span>
<br>
<span>template &lt;typename T1,</span>
<span>&nbsp;&nbsp;typename T2&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__pair.html#function-operator<=">thrust::operator&lt;=</a></b>(const pair< T1, T2 > & x,</span>
<span>&nbsp;&nbsp;const pair< T1, T2 > & y);</span>
<br>
<span>template &lt;typename T1,</span>
<span>&nbsp;&nbsp;typename T2&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__pair.html#function-operator>=">thrust::operator&gt;=</a></b>(const pair< T1, T2 > & x,</span>
<span>&nbsp;&nbsp;const pair< T1, T2 > & y);</span>
<br>
<span>template &lt;typename T1,</span>
<span>&nbsp;&nbsp;typename T2&gt;</span>
<span>__host__ __device__ void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__pair.html#function-swap">thrust::swap</a></b>(pair< T1, T2 > & x,</span>
<span>&nbsp;&nbsp;pair< T1, T2 > & y);</span>
<br>
<span>template &lt;typename T1,</span>
<span>&nbsp;&nbsp;typename T2&gt;</span>
<span>__host__ __device__ pair< T1, T2 > </span><span><b><a href="{{ site.baseurl }}/api/groups/group__pair.html#function-make-pair">thrust::make&#95;pair</a></b>(T1 x,</span>
<span>&nbsp;&nbsp;T2 y);</span>
</code>

## Member Classes

<h3 id="struct-thrustpair">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">Struct <code>thrust::pair</code>
</a>
</h3>

<h3 id="struct-thrusttuple-element">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1tuple__element.html">Struct <code>thrust::tuple&#95;element</code>
</a>
</h3>

<h3 id="struct-thrusttuple-size">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1tuple__size.html">Struct <code>thrust::tuple&#95;size</code>
</a>
</h3>


## Functions

<h3 id="function-operator==">
Function <code>thrust::operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T1,</span>
<span>&nbsp;&nbsp;typename T2&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const pair< T1, T2 > & x,</span>
<span>&nbsp;&nbsp;const pair< T1, T2 > & y);</span></code>
This operator tests two <code>pairs</code> for equality.

**Template Parameters**:
* **`T1`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>. 
* **`T2`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>. 

**Function Parameters**:
* **`x`** The first <code>pair</code> to compare. 
* **`y`** The second <code>pair</code> to compare. 

**Returns**:
<code>true</code> if and only if <code>x.first == y.first && x.second == y.second</code>.

<h3 id="function-operator<">
Function <code>thrust::operator&lt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T1,</span>
<span>&nbsp;&nbsp;typename T2&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator<</b>(const pair< T1, T2 > & x,</span>
<span>&nbsp;&nbsp;const pair< T1, T2 > & y);</span></code>
This operator tests two pairs for ascending ordering.

**Template Parameters**:
* **`T1`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>. 
* **`T2`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>. 

**Function Parameters**:
* **`x`** The first <code>pair</code> to compare. 
* **`y`** The second <code>pair</code> to compare. 

**Returns**:
<code>true</code> if and only if <code>x.first &lt; y.first || (!(y.first &lt; x.first) && x.second &lt; y.second)</code>.

<h3 id="function-operator!=">
Function <code>thrust::operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T1,</span>
<span>&nbsp;&nbsp;typename T2&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const pair< T1, T2 > & x,</span>
<span>&nbsp;&nbsp;const pair< T1, T2 > & y);</span></code>
This operator tests two pairs for inequality.

**Template Parameters**:
* **`T1`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>. 
* **`T2`** is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>. 

**Function Parameters**:
* **`x`** The first <code>pair</code> to compare. 
* **`y`** The second <code>pair</code> to compare. 

**Returns**:
<code>true</code> if and only if <code>!(x == y)</code>.

<h3 id="function-operator>">
Function <code>thrust::operator&gt;</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T1,</span>
<span>&nbsp;&nbsp;typename T2&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator></b>(const pair< T1, T2 > & x,</span>
<span>&nbsp;&nbsp;const pair< T1, T2 > & y);</span></code>
This operator tests two pairs for descending ordering.

**Template Parameters**:
* **`T1`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>. 
* **`T2`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>. 

**Function Parameters**:
* **`x`** The first <code>pair</code> to compare. 
* **`y`** The second <code>pair</code> to compare. 

**Returns**:
<code>true</code> if and only if <code>y &lt; x</code>.

<h3 id="function-operator<=">
Function <code>thrust::operator&lt;=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T1,</span>
<span>&nbsp;&nbsp;typename T2&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator<=</b>(const pair< T1, T2 > & x,</span>
<span>&nbsp;&nbsp;const pair< T1, T2 > & y);</span></code>
This operator tests two pairs for ascending ordering or equivalence.

**Template Parameters**:
* **`T1`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>. 
* **`T2`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>. 

**Function Parameters**:
* **`x`** The first <code>pair</code> to compare. 
* **`y`** The second <code>pair</code> to compare. 

**Returns**:
<code>true</code> if and only if <code>!(y &lt; x)</code>.

<h3 id="function-operator>=">
Function <code>thrust::operator&gt;=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T1,</span>
<span>&nbsp;&nbsp;typename T2&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator>=</b>(const pair< T1, T2 > & x,</span>
<span>&nbsp;&nbsp;const pair< T1, T2 > & y);</span></code>
This operator tests two pairs for descending ordering or equivalence.

**Template Parameters**:
* **`T1`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>. 
* **`T2`** is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>. 

**Function Parameters**:
* **`x`** The first <code>pair</code> to compare. 
* **`y`** The second <code>pair</code> to compare. 

**Returns**:
<code>true</code> if and only if <code>!(x &lt; y)</code>.

<h3 id="function-swap">
Function <code>thrust::swap</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T1,</span>
<span>&nbsp;&nbsp;typename T2&gt;</span>
<span>__host__ __device__ void </span><span><b>swap</b>(pair< T1, T2 > & x,</span>
<span>&nbsp;&nbsp;pair< T1, T2 > & y);</span></code>
<code>swap</code> swaps the contents of two <code>pair</code>s.

**Function Parameters**:
* **`x`** The first <code>pair</code> to swap. 
* **`y`** The second <code>pair</code> to swap. 

<h3 id="function-make-pair">
Function <code>thrust::make&#95;pair</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T1,</span>
<span>&nbsp;&nbsp;typename T2&gt;</span>
<span>__host__ __device__ pair< T1, T2 > </span><span><b>make_pair</b>(T1 x,</span>
<span>&nbsp;&nbsp;T2 y);</span></code>
This convenience function creates a <code>pair</code> from two objects.

**Template Parameters**:
* **`T1`** There are no requirements on the type of <code>T1</code>. 
* **`T2`** There are no requirements on the type of <code>T2</code>. 

**Function Parameters**:
* **`x`** The first object to copy from. 
* **`y`** The second object to copy from. 

**Returns**:
A newly-constructed <code>pair</code> copied from <code>a</code> and <code>b</code>.


