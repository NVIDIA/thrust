---
title: thrust::mr::polymorphic_adaptor_resource
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::mr::polymorphic_adaptor_resource`

**Inherits From**:
[`thrust::mr::memory_resource< void * >`]({{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html)

<code class="doxybook">
<span>template &lt;typename Pointer = void &#42;&gt;</span>
<span>class thrust::mr::polymorphic&#95;adaptor&#95;resource {</span>
<span>public:</span><span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">thrust::mr::memory&#95;resource&lt; void &#42; &gt;</a></b></code> */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#typedef-pointer">pointer</a></b>;</span>
<br>
<span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1polymorphic__adaptor__resource.html#function-polymorphic-adaptor-resource">polymorphic&#95;adaptor&#95;resource</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">memory_resource</a>< Pointer > * t);</span>
<br>
<span>&nbsp;&nbsp;virtual Pointer </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1polymorphic__adaptor__resource.html#function-do-allocate">do&#95;allocate</a></b>(std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t)) override;</span>
<br>
<span>&nbsp;&nbsp;virtual void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1polymorphic__adaptor__resource.html#function-do-deallocate">do&#95;deallocate</a></b>(Pointer p,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment) override;</span>
<br>
<span>&nbsp;&nbsp;virtual __host__ virtual __device__ bool </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1polymorphic__adaptor__resource.html#function-do-is-equal">do&#95;is&#95;equal</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">memory_resource</a>< Pointer > & other) const override;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">thrust::mr::memory&#95;resource&lt; void &#42; &gt;</a></b></code> */</span><span>&nbsp;&nbsp;virtual </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#function-~memory-resource">~memory&#95;resource</a></b>() = default;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">thrust::mr::memory&#95;resource&lt; void &#42; &gt;</a></b></code> */</span><span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#typedef-pointer">pointer</a> </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#function-allocate">allocate</a></b>(std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t));</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">thrust::mr::memory&#95;resource&lt; void &#42; &gt;</a></b></code> */</span><span>&nbsp;&nbsp;void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#function-deallocate">deallocate</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#typedef-pointer">pointer</a> p,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t));</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">thrust::mr::memory&#95;resource&lt; void &#42; &gt;</a></b></code> */</span><span>&nbsp;&nbsp;__host__ __device__ bool </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#function-is-equal">is&#95;equal</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">memory_resource</a> & other) const;</span>
<span>};</span>
</code>

## Member Functions

<h3 id="function-polymorphic-adaptor-resource">
Function <code>thrust::mr::polymorphic&#95;adaptor&#95;resource::polymorphic&#95;adaptor&#95;resource</code>
</h3>

<code class="doxybook">
<span><b>polymorphic_adaptor_resource</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">memory_resource</a>< Pointer > * t);</span></code>
<h3 id="function-do-allocate">
Function <code>thrust::mr::polymorphic&#95;adaptor&#95;resource::do&#95;allocate</code>
</h3>

<code class="doxybook">
<span>virtual Pointer </span><span><b>do_allocate</b>(std::size_t bytes,</span>
<span>&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t)) override;</span></code>
<h3 id="function-do-deallocate">
Function <code>thrust::mr::polymorphic&#95;adaptor&#95;resource::do&#95;deallocate</code>
</h3>

<code class="doxybook">
<span>virtual void </span><span><b>do_deallocate</b>(Pointer p,</span>
<span>&nbsp;&nbsp;std::size_t bytes,</span>
<span>&nbsp;&nbsp;std::size_t alignment) override;</span></code>
<h3 id="function-do-is-equal">
Function <code>thrust::mr::polymorphic&#95;adaptor&#95;resource::do&#95;is&#95;equal</code>
</h3>

<code class="doxybook">
<span>virtual __host__ virtual __device__ bool </span><span><b>do_is_equal</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">memory_resource</a>< Pointer > & other) const override;</span></code>

