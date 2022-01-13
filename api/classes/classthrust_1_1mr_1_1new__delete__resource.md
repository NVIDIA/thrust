---
title: thrust::mr::new_delete_resource
parent: Memory Resources
grand_parent: Memory Management
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::mr::new_delete_resource`

A memory resource that uses global operators new and delete to allocate and deallocate memory. Uses alignment-enabled overloads when available, otherwise uses regular overloads and implements alignment requirements by itself. 

**Inherits From**:
[`thrust::mr::memory_resource<>`]({{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html)

<code class="doxybook">
<span>#include <thrust/mr/new.h></span><br>
<span>class thrust::mr::new&#95;delete&#95;resource {</span>
<span>public:</span><span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">thrust::mr::memory&#95;resource&lt;&gt;</a></b></code> */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#typedef-pointer">pointer</a></b>;</span>
<br>
<span>&nbsp;&nbsp;virtual void * </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1new__delete__resource.html#function-do-allocate">do&#95;allocate</a></b>(std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t)) override;</span>
<br>
<span>&nbsp;&nbsp;void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1new__delete__resource.html#function-do-deallocate">do&#95;deallocate</a></b>(void * p,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t)) override;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">thrust::mr::memory&#95;resource&lt;&gt;</a></b></code> */</span><span>&nbsp;&nbsp;virtual </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#function-~memory-resource">~memory&#95;resource</a></b>() = default;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">thrust::mr::memory&#95;resource&lt;&gt;</a></b></code> */</span><span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#typedef-pointer">pointer</a> </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#function-allocate">allocate</a></b>(std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t));</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">thrust::mr::memory&#95;resource&lt;&gt;</a></b></code> */</span><span>&nbsp;&nbsp;void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#function-deallocate">deallocate</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#typedef-pointer">pointer</a> p,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t));</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">thrust::mr::memory&#95;resource&lt;&gt;</a></b></code> */</span><span>&nbsp;&nbsp;__host__ __device__ bool </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#function-is-equal">is&#95;equal</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">memory_resource</a> & other) const;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">thrust::mr::memory&#95;resource&lt;&gt;</a></b></code> */</span><span>&nbsp;&nbsp;virtual __host__ virtual __device__ bool </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#function-do-is-equal">do&#95;is&#95;equal</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">memory_resource</a> & other) const;</span>
<span>};</span>
</code>

## Member Functions

<h3 id="function-do-allocate">
Function <code>thrust::mr::new&#95;delete&#95;resource::do&#95;allocate</code>
</h3>

<code class="doxybook">
<span>virtual void * </span><span><b>do_allocate</b>(std::size_t bytes,</span>
<span>&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t)) override;</span></code>
Allocates memory of size at least <code>bytes</code> and alignment at least <code>alignment</code>.

**Function Parameters**:
* **`bytes`** size, in bytes, that is requested from this allocation 
* **`alignment`** alignment that is requested from this allocation 

**Returns**:
A pointer to void to the newly allocated memory. 

**Exceptions**:
**`thrust::bad_alloc`**: when no memory with requested size and alignment can be allocated. 

**Implements**: [`do_allocate`]({{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#function-do-allocate)

<h3 id="function-do-deallocate">
Function <code>thrust::mr::new&#95;delete&#95;resource::do&#95;deallocate</code>
</h3>

<code class="doxybook">
<span>void </span><span><b>do_deallocate</b>(void * p,</span>
<span>&nbsp;&nbsp;std::size_t bytes,</span>
<span>&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t)) override;</span></code>

