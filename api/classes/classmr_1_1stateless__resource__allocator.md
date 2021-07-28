---
title: mr::stateless_resource_allocator
parent: Allocators
grand_parent: Memory Management
nav_exclude: true
has_children: true
has_toc: false
---

# Class `mr::stateless_resource_allocator`

A helper allocator class that uses global instances of a given upstream memory resource. Requires the memory resource to be default constructible.

**Template Parameters**:
* **`T`** the type that will be allocated by this allocator. 
* **`Upstream`** the upstream memory resource to use for memory allocation. Must derive from <code>thrust::mr::memory&#95;resource</code> and must be <code>final</code> (in C++11 and beyond). 

**Inherits From**:
`thrust::mr::allocator< T, Upstream >`

<code class="doxybook">
<span>#include <thrust/mr/allocator.h></span><br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename Upstream&gt;</span>
<span>class mr::stateless&#95;resource&#95;allocator {</span>
<span>public:</span><span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;struct <b><a href="/api/classes/structmr_1_1stateless__resource__allocator_1_1rebind.html">rebind</a></b>;</span>
<br>
<span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classmr_1_1stateless__resource__allocator.html#function-stateless_resource_allocator">stateless&#95;resource&#95;allocator</a></b>();</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classmr_1_1stateless__resource__allocator.html#function-stateless_resource_allocator">stateless&#95;resource&#95;allocator</a></b>(const <a href="/api/classes/classmr_1_1stateless__resource__allocator.html">stateless_resource_allocator</a> & other);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classmr_1_1stateless__resource__allocator.html#function-stateless_resource_allocator">stateless&#95;resource&#95;allocator</a></b>(const <a href="/api/classes/classmr_1_1stateless__resource__allocator.html">stateless_resource_allocator</a>< U, Upstream > & other);</span>
<br>
<span>&nbsp;&nbsp;<a href="/api/classes/classmr_1_1stateless__resource__allocator.html">stateless_resource_allocator</a> & </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classmr_1_1stateless__resource__allocator.html#function-operator=">operator=</a></b>(const <a href="/api/classes/classmr_1_1stateless__resource__allocator.html">stateless_resource_allocator</a> &) = default;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classmr_1_1stateless__resource__allocator.html#function-~stateless_resource_allocator">~stateless&#95;resource&#95;allocator</a></b>();</span>
<span>};</span>
</code>

## Member Classes

<h3 id="struct-mr::stateless_resource_allocator::rebind">
<a href="/api/classes/structmr_1_1stateless__resource__allocator_1_1rebind.html">Struct <code>mr::stateless&#95;resource&#95;allocator::mr::stateless&#95;resource&#95;allocator::rebind</code>
</a>
</h3>


## Member Functions

<h3 id="function-stateless_resource_allocator">
Function <code>mr::stateless&#95;resource&#95;allocator::&gt;::stateless&#95;resource&#95;allocator</code>
</h3>

<code class="doxybook">
<span>__host__ </span><span><b>stateless_resource_allocator</b>();</span></code>
Default constructor. Uses <code>get&#95;global&#95;resource</code> to get the global instance of <code>Upstream</code> and initializes the <code>allocator</code> base subobject with that resource. 

<h3 id="function-stateless_resource_allocator">
Function <code>mr::stateless&#95;resource&#95;allocator::&gt;::stateless&#95;resource&#95;allocator</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>stateless_resource_allocator</b>(const <a href="/api/classes/classmr_1_1stateless__resource__allocator.html">stateless_resource_allocator</a> & other);</span></code>
Copy constructor. Copies the memory resource pointer. 

<h3 id="function-stateless_resource_allocator">
Function <code>mr::stateless&#95;resource&#95;allocator::&gt;::stateless&#95;resource&#95;allocator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ </span><span><b>stateless_resource_allocator</b>(const <a href="/api/classes/classmr_1_1stateless__resource__allocator.html">stateless_resource_allocator</a>< U, Upstream > & other);</span></code>
Conversion constructor from an allocator of a different type. Copies the memory resource pointer. 

<h3 id="function-operator=">
Function <code>mr::stateless&#95;resource&#95;allocator::&gt;::operator=</code>
</h3>

<code class="doxybook">
<span><a href="/api/classes/classmr_1_1stateless__resource__allocator.html">stateless_resource_allocator</a> & </span><span><b>operator=</b>(const <a href="/api/classes/classmr_1_1stateless__resource__allocator.html">stateless_resource_allocator</a> &) = default;</span></code>
<h3 id="function-~stateless_resource_allocator">
Function <code>mr::stateless&#95;resource&#95;allocator::&gt;::~stateless&#95;resource&#95;allocator</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>~stateless_resource_allocator</b>();</span></code>
Destructor. 


