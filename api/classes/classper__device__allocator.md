---
title: per_device_allocator
nav_exclude: true
has_children: true
has_toc: false
---

# Class `per_device_allocator`

A helper allocator class that uses global per device instances of a given upstream memory resource. Requires the memory resource to be default constructible.

**Template Parameters**:
* **`T`** the type that will be allocated by this allocator. 
* **`MR`** the upstream memory resource to use for memory allocation. Must derive from <code>thrust::mr::memory&#95;resource</code> and must be <code>final</code>. 
* **`ExecutionPolicy`** the execution policy of the system to be used to retrieve the resource for the current device. 

**Inherits From**:
`thrust::mr::allocator< T, Upstream >`

<code class="doxybook">
<span>#include <thrust/per_device_resource.h></span><br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename Upstream,</span>
<span>&nbsp;&nbsp;typename ExecutionPolicy&gt;</span>
<span>class per&#95;device&#95;allocator {</span>
<span>public:</span><span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;struct <b><a href="/thrust/api/classes/structper__device__allocator_1_1rebind.html">rebind</a></b>;</span>
<br>
<span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classper__device__allocator.html#function-per_device_allocator">per&#95;device&#95;allocator</a></b>();</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classper__device__allocator.html#function-per_device_allocator">per&#95;device&#95;allocator</a></b>(const <a href="/thrust/api/classes/classper__device__allocator.html">per_device_allocator</a> & other);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classper__device__allocator.html#function-per_device_allocator">per&#95;device&#95;allocator</a></b>(const <a href="/thrust/api/classes/classper__device__allocator.html">per_device_allocator</a>< U, Upstream, ExecutionPolicy > & other);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classper__device__allocator.html#function-~per_device_allocator">~per&#95;device&#95;allocator</a></b>();</span>
<span>};</span>
</code>

## Member Classes

<h3 id="struct-per_device_allocator::rebind">
<a href="/thrust/api/classes/structper__device__allocator_1_1rebind.html">Struct <code>per&#95;device&#95;allocator::per&#95;device&#95;allocator::rebind</code>
</a>
</h3>


## Member Functions

<h3 id="function-per_device_allocator">
Function <code>per&#95;device&#95;allocator::&gt;::per&#95;device&#95;allocator</code>
</h3>

<code class="doxybook">
<span>__host__ </span><span><b>per_device_allocator</b>();</span></code>
Default constructor. Uses <code>get&#95;global&#95;resource</code> to get the global instance of <code>Upstream</code> and initializes the <code>allocator</code> base subobject with that resource. 

<h3 id="function-per_device_allocator">
Function <code>per&#95;device&#95;allocator::&gt;::per&#95;device&#95;allocator</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>per_device_allocator</b>(const <a href="/thrust/api/classes/classper__device__allocator.html">per_device_allocator</a> & other);</span></code>
Copy constructor. Copies the memory resource pointer. 

<h3 id="function-per_device_allocator">
Function <code>per&#95;device&#95;allocator::&gt;::per&#95;device&#95;allocator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ </span><span><b>per_device_allocator</b>(const <a href="/thrust/api/classes/classper__device__allocator.html">per_device_allocator</a>< U, Upstream, ExecutionPolicy > & other);</span></code>
Conversion constructor from an allocator of a different type. Copies the memory resource pointer. 

<h3 id="function-~per_device_allocator">
Function <code>per&#95;device&#95;allocator::&gt;::~per&#95;device&#95;allocator</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>~per_device_allocator</b>();</span></code>
Destructor. 


