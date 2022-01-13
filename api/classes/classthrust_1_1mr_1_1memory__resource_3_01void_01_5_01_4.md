---
title: thrust::mr::memory_resource< void * >
parent: Memory Resources
grand_parent: Memory Management
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::mr::memory_resource< void * >`

<code class="doxybook">
<span>class thrust::mr::memory&#95;resource&lt; void &#42; &gt; {</span>
<span>public:</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource_3_01void_01_5_01_4.html#typedef-pointer">pointer</a></b>;</span>
<br>
<span>&nbsp;&nbsp;virtual </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource_3_01void_01_5_01_4.html#function-~memory-resource">~memory&#95;resource</a></b>() = default;</span>
<br>
<span>&nbsp;&nbsp;pointer </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource_3_01void_01_5_01_4.html#function-allocate">allocate</a></b>(std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t));</span>
<br>
<span>&nbsp;&nbsp;void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource_3_01void_01_5_01_4.html#function-deallocate">deallocate</a></b>(pointer p,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t));</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ bool </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource_3_01void_01_5_01_4.html#function-is-equal">is&#95;equal</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">memory_resource</a> & other) const;</span>
<br>
<span>&nbsp;&nbsp;virtual pointer </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource_3_01void_01_5_01_4.html#function-do-allocate">do&#95;allocate</a></b>(std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment) = 0;</span>
<br>
<span>&nbsp;&nbsp;virtual void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource_3_01void_01_5_01_4.html#function-do-deallocate">do&#95;deallocate</a></b>(pointer p,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment) = 0;</span>
<br>
<span>&nbsp;&nbsp;virtual __host__ virtual __device__ bool </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource_3_01void_01_5_01_4.html#function-do-is-equal">do&#95;is&#95;equal</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">memory_resource</a> & other) const;</span>
<span>};</span>
</code>

## Member Types

<h3 id="typedef-pointer">
Typedef <code>thrust::mr::memory&#95;resource&lt; void &#42; &gt;::pointer</code>
</h3>

<code class="doxybook">
<span>typedef void *<b>pointer</b>;</span></code>

## Member Functions

<h3 id="function-~memory-resource">
Function <code>thrust::mr::memory&#95;resource&lt; void &#42; &gt;::~memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>virtual </span><span><b>~memory_resource</b>() = default;</span></code>
<h3 id="function-allocate">
Function <code>thrust::mr::memory&#95;resource&lt; void &#42; &gt;::allocate</code>
</h3>

<code class="doxybook">
<span>pointer </span><span><b>allocate</b>(std::size_t bytes,</span>
<span>&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t));</span></code>
<h3 id="function-deallocate">
Function <code>thrust::mr::memory&#95;resource&lt; void &#42; &gt;::deallocate</code>
</h3>

<code class="doxybook">
<span>void </span><span><b>deallocate</b>(pointer p,</span>
<span>&nbsp;&nbsp;std::size_t bytes,</span>
<span>&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t));</span></code>
<h3 id="function-is-equal">
Function <code>thrust::mr::memory&#95;resource&lt; void &#42; &gt;::is&#95;equal</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ bool </span><span><b>is_equal</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">memory_resource</a> & other) const;</span></code>
<h3 id="function-do-allocate">
Function <code>thrust::mr::memory&#95;resource&lt; void &#42; &gt;::do&#95;allocate</code>
</h3>

<code class="doxybook">
<span>virtual pointer </span><span><b>do_allocate</b>(std::size_t bytes,</span>
<span>&nbsp;&nbsp;std::size_t alignment) = 0;</span></code>
<h3 id="function-do-deallocate">
Function <code>thrust::mr::memory&#95;resource&lt; void &#42; &gt;::do&#95;deallocate</code>
</h3>

<code class="doxybook">
<span>virtual void </span><span><b>do_deallocate</b>(pointer p,</span>
<span>&nbsp;&nbsp;std::size_t bytes,</span>
<span>&nbsp;&nbsp;std::size_t alignment) = 0;</span></code>
<h3 id="function-do-is-equal">
Function <code>thrust::mr::memory&#95;resource&lt; void &#42; &gt;::do&#95;is&#95;equal</code>
</h3>

<code class="doxybook">
<span>virtual __host__ virtual __device__ bool </span><span><b>do_is_equal</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">memory_resource</a> & other) const;</span></code>

