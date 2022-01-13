---
title: Allocators
parent: Memory Management
grand_parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Allocators

<code class="doxybook">
<span>template &lt;typename Upstream&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr__memory__resource.html">thrust::device&#95;ptr&#95;memory&#95;resource</a></b>;</span>
<br>
<span class="doxybook-comment">/* An allocator which creates new elements in memory accessible by devices.  */</span><span>template &lt;typename T&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__allocator.html">thrust::device&#95;allocator</a></b>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__malloc__allocator.html">thrust::device&#95;malloc&#95;allocator</a></b>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__new__allocator.html">thrust::device&#95;new&#95;allocator</a></b>;</span>
<br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;class MR&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">thrust::mr::allocator</a></b>;</span>
<br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename Upstream&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1stateless__resource__allocator.html">thrust::mr::stateless&#95;resource&#95;allocator</a></b>;</span>
<br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename Pointer&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__allocators.html#using-polymorphic-allocator">thrust::mr::polymorphic&#95;allocator</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename MR&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__allocators.html#function-operator==">thrust::mr::operator==</a></b>(const allocator< T, MR > & lhs,</span>
<span>&nbsp;&nbsp;const allocator< T, MR > & rhs);</span>
<br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename MR&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__allocators.html#function-operator!=">thrust::mr::operator!=</a></b>(const allocator< T, MR > & lhs,</span>
<span>&nbsp;&nbsp;const allocator< T, MR > & rhs);</span>
</code>

## Member Classes

<h3 id="class-thrustdevice-ptr-memory-resource">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr__memory__resource.html">Class <code>thrust::device&#95;ptr&#95;memory&#95;resource</code>
</a>
</h3>

**Inherits From**:
[`thrust::mr::memory_resource< device_ptr< void > >`]({{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html)

<h3 id="class-thrustdevice-allocator">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__allocator.html">Class <code>thrust::device&#95;allocator</code>
</a>
</h3>

An allocator which creates new elements in memory accessible by devices. 

**Inherits From**:
* `thrust::mr::stateless_resource_allocator< T, device_ptr_memory_resource< device_memory_resource > >`
* `thrust::mr::allocator< T, Upstream >`
* `thrust::mr::validator< MR >`

<h3 id="class-thrustdevice-malloc-allocator">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__malloc__allocator.html">Class <code>thrust::device&#95;malloc&#95;allocator</code>
</a>
</h3>

<h3 id="class-thrustdevice-new-allocator">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__new__allocator.html">Class <code>thrust::device&#95;new&#95;allocator</code>
</a>
</h3>

<h3 id="class-thrustmrallocator">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">Class <code>thrust::mr::allocator</code>
</a>
</h3>

**Inherits From**:
`thrust::mr::validator< MR >`

<h3 id="class-thrustmrstateless-resource-allocator">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1stateless__resource__allocator.html">Class <code>thrust::mr::stateless&#95;resource&#95;allocator</code>
</a>
</h3>

**Inherits From**:
* `thrust::mr::allocator< T, Upstream >`
* `thrust::mr::validator< MR >`


## Types

<h3 id="using-polymorphic-allocator">
Type Alias <code>thrust::mr::polymorphic&#95;allocator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename Pointer&gt;</span>
<span>using <b>polymorphic_allocator</b> = allocator&lt; T, polymorphic&#95;adaptor&#95;resource&lt; Pointer &gt; &gt;;</span></code>

## Functions

<h3 id="function-operator==">
Function <code>thrust::mr::operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename MR&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const allocator< T, MR > & lhs,</span>
<span>&nbsp;&nbsp;const allocator< T, MR > & rhs);</span></code>
Compares the allocators for equality by comparing the underlying memory resources. 

<h3 id="function-operator!=">
Function <code>thrust::mr::operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename MR&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const allocator< T, MR > & lhs,</span>
<span>&nbsp;&nbsp;const allocator< T, MR > & rhs);</span></code>
Compares the allocators for inequality by comparing the underlying memory resources. 


