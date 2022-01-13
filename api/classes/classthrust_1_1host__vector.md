---
title: thrust::host_vector
parent: Host Containers
grand_parent: Container Classes
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::host_vector`

A <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host&#95;vector</a></code> is a container that supports random access to elements, constant time removal of elements at the end, and linear time insertion and removal of elements at the beginning or in the middle. The number of elements in a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host&#95;vector</a></code> may vary dynamically; memory management is automatic. The memory associated with a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host&#95;vector</a></code> resides in memory accessible to hosts.

**Inherits From**:
`detail::vector_base< T, std::allocator< T > >`

**See**:
* <a href="https://en.cppreference.com/w/cpp/container/vector">https://en.cppreference.com/w/cpp/container/vector</a>
* <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device_vector</a>
* universal_vector 

<code class="doxybook">
<span>#include <thrust/host_vector.h></span><br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename Alloc = std::allocator&lt;T&gt;&gt;</span>
<span>class thrust::host&#95;vector {</span>
<span>public:</span><span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html#function-host-vector">host&#95;vector</a></b>(void);</span>
<br>
<span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html#function-host-vector">host&#95;vector</a></b>(const Alloc & alloc);</span>
<br>
<span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html#function-~host-vector">~host&#95;vector</a></b>(void);</span>
<br>
<span>&nbsp;&nbsp;explicit __host__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html#function-host-vector">host&#95;vector</a></b>(size_type n);</span>
<br>
<span>&nbsp;&nbsp;explicit __host__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html#function-host-vector">host&#95;vector</a></b>(size_type n,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const Alloc & alloc);</span>
<br>
<span>&nbsp;&nbsp;explicit __host__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html#function-host-vector">host&#95;vector</a></b>(size_type n,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const value_type & value);</span>
<br>
<span>&nbsp;&nbsp;explicit __host__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html#function-host-vector">host&#95;vector</a></b>(size_type n,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const value_type & value,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const Alloc & alloc);</span>
<br>
<span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html#function-host-vector">host&#95;vector</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> & v);</span>
<br>
<span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html#function-host-vector">host&#95;vector</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> & v,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const Alloc & alloc);</span>
<br>
<span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html#function-host-vector">host&#95;vector</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> && v);</span>
<br>
<span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html#function-host-vector">host&#95;vector</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> && v,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const Alloc & alloc);</span>
<br>
<span>&nbsp;&nbsp;__host__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html#function-operator=">operator=</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> & v);</span>
<br>
<span>&nbsp;&nbsp;__host__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html#function-operator=">operator=</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> && v);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename OtherT,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typename OtherAlloc&gt;</span>
<span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html#function-host-vector">host&#95;vector</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a>< OtherT, OtherAlloc > & v);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename OtherT,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typename OtherAlloc&gt;</span>
<span>&nbsp;&nbsp;__host__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html#function-operator=">operator=</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a>< OtherT, OtherAlloc > & v);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename OtherT,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typename OtherAlloc&gt;</span>
<span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html#function-host-vector">host&#95;vector</a></b>(const std::vector< OtherT, OtherAlloc > & v);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename OtherT,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typename OtherAlloc&gt;</span>
<span>&nbsp;&nbsp;__host__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html#function-operator=">operator=</a></b>(const std::vector< OtherT, OtherAlloc > & v);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename OtherT,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typename OtherAlloc&gt;</span>
<span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html#function-host-vector">host&#95;vector</a></b>(const detail::vector_base< OtherT, OtherAlloc > & v);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename OtherT,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typename OtherAlloc&gt;</span>
<span>&nbsp;&nbsp;__host__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html#function-operator=">operator=</a></b>(const detail::vector_base< OtherT, OtherAlloc > & v);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename InputIterator&gt;</span>
<span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html#function-host-vector">host&#95;vector</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;InputIterator last);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename InputIterator&gt;</span>
<span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html#function-host-vector">host&#95;vector</a></b>(InputIterator first,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;const Alloc & alloc);</span>
<span>};</span>
</code>

## Member Functions

<h3 id="function-host-vector">
Function <code>thrust::host&#95;vector::host&#95;vector</code>
</h3>

<code class="doxybook">
<span>__host__ </span><span><b>host_vector</b>(void);</span></code>
This constructor creates an empty <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host&#95;vector</a></code>. 

<h3 id="function-host-vector">
Function <code>thrust::host&#95;vector::host&#95;vector</code>
</h3>

<code class="doxybook">
<span>__host__ </span><span><b>host_vector</b>(const Alloc & alloc);</span></code>
This constructor creates an empty <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host&#95;vector</a></code>. 

**Function Parameters**:
**`alloc`**: The allocator to use by this <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a>. 

<h3 id="function-~host-vector">
Function <code>thrust::host&#95;vector::~host&#95;vector</code>
</h3>

<code class="doxybook">
<span>__host__ </span><span><b>~host_vector</b>(void);</span></code>
The destructor erases the elements. 

<h3 id="function-host-vector">
Function <code>thrust::host&#95;vector::host&#95;vector</code>
</h3>

<code class="doxybook">
<span>explicit __host__ </span><span><b>host_vector</b>(size_type n);</span></code>
This constructor creates a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host&#95;vector</a></code> with the given size. 

**Function Parameters**:
**`n`**: The number of elements to initially create. 

<h3 id="function-host-vector">
Function <code>thrust::host&#95;vector::host&#95;vector</code>
</h3>

<code class="doxybook">
<span>explicit __host__ </span><span><b>host_vector</b>(size_type n,</span>
<span>&nbsp;&nbsp;const Alloc & alloc);</span></code>
This constructor creates a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host&#95;vector</a></code> with the given size. 

**Function Parameters**:
* **`n`** The number of elements to initially create. 
* **`alloc`** The allocator to use by this <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a>. 

<h3 id="function-host-vector">
Function <code>thrust::host&#95;vector::host&#95;vector</code>
</h3>

<code class="doxybook">
<span>explicit __host__ </span><span><b>host_vector</b>(size_type n,</span>
<span>&nbsp;&nbsp;const value_type & value);</span></code>
This constructor creates a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host&#95;vector</a></code> with copies of an exemplar element. 

**Function Parameters**:
* **`n`** The number of elements to initially create. 
* **`value`** An element to copy. 

<h3 id="function-host-vector">
Function <code>thrust::host&#95;vector::host&#95;vector</code>
</h3>

<code class="doxybook">
<span>explicit __host__ </span><span><b>host_vector</b>(size_type n,</span>
<span>&nbsp;&nbsp;const value_type & value,</span>
<span>&nbsp;&nbsp;const Alloc & alloc);</span></code>
This constructor creates a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host&#95;vector</a></code> with copies of an exemplar element. 

**Function Parameters**:
* **`n`** The number of elements to initially create. 
* **`value`** An element to copy. 
* **`alloc`** The allocator to use by this <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a>. 

<h3 id="function-host-vector">
Function <code>thrust::host&#95;vector::host&#95;vector</code>
</h3>

<code class="doxybook">
<span>__host__ </span><span><b>host_vector</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> & v);</span></code>
Copy constructor copies from an exemplar <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host&#95;vector</a></code>. 

**Function Parameters**:
**`v`**: The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host&#95;vector</a></code> to copy. 

<h3 id="function-host-vector">
Function <code>thrust::host&#95;vector::host&#95;vector</code>
</h3>

<code class="doxybook">
<span>__host__ </span><span><b>host_vector</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> & v,</span>
<span>&nbsp;&nbsp;const Alloc & alloc);</span></code>
Copy constructor copies from an exemplar <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host&#95;vector</a></code>. 

**Function Parameters**:
* **`v`** The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host&#95;vector</a></code> to copy. 
* **`alloc`** The allocator to use by this <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a>. 

<h3 id="function-host-vector">
Function <code>thrust::host&#95;vector::host&#95;vector</code>
</h3>

<code class="doxybook">
<span>__host__ </span><span><b>host_vector</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> && v);</span></code>
Move constructor moves from another <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a>. 

**Function Parameters**:
**`v`**: The <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> to move. 

<h3 id="function-host-vector">
Function <code>thrust::host&#95;vector::host&#95;vector</code>
</h3>

<code class="doxybook">
<span>__host__ </span><span><b>host_vector</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> && v,</span>
<span>&nbsp;&nbsp;const Alloc & alloc);</span></code>
Move constructor moves from another <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a>. 

**Function Parameters**:
* **`v`** The <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> to move. 
* **`alloc`** The allocator to use by this <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a>. 

<h3 id="function-operator=">
Function <code>thrust::host&#95;vector::operator=</code>
</h3>

<code class="doxybook">
<span>__host__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> & </span><span><b>operator=</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> & v);</span></code>
Assign operator copies from an exemplar <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host&#95;vector</a></code>. 

**Function Parameters**:
**`v`**: The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host&#95;vector</a></code> to copy. 

<h3 id="function-operator=">
Function <code>thrust::host&#95;vector::operator=</code>
</h3>

<code class="doxybook">
<span>__host__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> & </span><span><b>operator=</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> && v);</span></code>
Move assign operator moves from another <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a>. 

**Function Parameters**:
**`v`**: The <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> to move. 

<h3 id="function-host-vector">
Function <code>thrust::host&#95;vector::host&#95;vector</code>
</h3>

<code class="doxybook">
<span>template &lt;typename OtherT,</span>
<span>&nbsp;&nbsp;typename OtherAlloc&gt;</span>
<span>__host__ </span><span><b>host_vector</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a>< OtherT, OtherAlloc > & v);</span></code>
Copy constructor copies from an exemplar <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host&#95;vector</a></code> with different type. 

**Function Parameters**:
**`v`**: The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host&#95;vector</a></code> to copy. 

<h3 id="function-operator=">
Function <code>thrust::host&#95;vector::operator=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename OtherT,</span>
<span>&nbsp;&nbsp;typename OtherAlloc&gt;</span>
<span>__host__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> & </span><span><b>operator=</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a>< OtherT, OtherAlloc > & v);</span></code>
Assign operator copies from an exemplar <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host&#95;vector</a></code> with different type. 

**Function Parameters**:
**`v`**: The <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host&#95;vector</a></code> to copy. 

<h3 id="function-host-vector">
Function <code>thrust::host&#95;vector::host&#95;vector</code>
</h3>

<code class="doxybook">
<span>template &lt;typename OtherT,</span>
<span>&nbsp;&nbsp;typename OtherAlloc&gt;</span>
<span>__host__ </span><span><b>host_vector</b>(const std::vector< OtherT, OtherAlloc > & v);</span></code>
Copy constructor copies from an exemplar <code>std::vector</code>. 

**Function Parameters**:
**`v`**: The <code>std::vector</code> to copy. 

<h3 id="function-operator=">
Function <code>thrust::host&#95;vector::operator=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename OtherT,</span>
<span>&nbsp;&nbsp;typename OtherAlloc&gt;</span>
<span>__host__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> & </span><span><b>operator=</b>(const std::vector< OtherT, OtherAlloc > & v);</span></code>
Assign operator copies from an exemplar <code>std::vector</code>. 

**Function Parameters**:
**`v`**: The <code>std::vector</code> to copy. 

<h3 id="function-host-vector">
Function <code>thrust::host&#95;vector::host&#95;vector</code>
</h3>

<code class="doxybook">
<span>template &lt;typename OtherT,</span>
<span>&nbsp;&nbsp;typename OtherAlloc&gt;</span>
<span>__host__ </span><span><b>host_vector</b>(const detail::vector_base< OtherT, OtherAlloc > & v);</span></code>
Copy construct from a <code>vector&#95;base</code> whose element type is convertible to <code>T</code>.

**Function Parameters**:
**`v`**: The <code>vector&#95;base</code> to copy. 

<h3 id="function-operator=">
Function <code>thrust::host&#95;vector::operator=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename OtherT,</span>
<span>&nbsp;&nbsp;typename OtherAlloc&gt;</span>
<span>__host__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> & </span><span><b>operator=</b>(const detail::vector_base< OtherT, OtherAlloc > & v);</span></code>
Assign a <code>vector&#95;base</code> whose element type is convertible to <code>T</code>.

**Function Parameters**:
**`v`**: The <code>vector&#95;base</code> to copy. 

<h3 id="function-host-vector">
Function <code>thrust::host&#95;vector::host&#95;vector</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator&gt;</span>
<span>__host__ </span><span><b>host_vector</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last);</span></code>
This constructor builds a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host&#95;vector</a></code> from a range. 

**Function Parameters**:
* **`first`** The beginning of the range. 
* **`last`** The end of the range. 

<h3 id="function-host-vector">
Function <code>thrust::host&#95;vector::host&#95;vector</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputIterator&gt;</span>
<span>__host__ </span><span><b>host_vector</b>(InputIterator first,</span>
<span>&nbsp;&nbsp;InputIterator last,</span>
<span>&nbsp;&nbsp;const Alloc & alloc);</span></code>
This constructor builds a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host&#95;vector</a></code> from a range. 

**Function Parameters**:
* **`first`** The beginning of the range. 
* **`last`** The end of the range. 
* **`alloc`** The allocator to use by this <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a>. 


