---
title: device_ptr_memory_resource
parent: Allocators
grand_parent: Memory Management
nav_exclude: true
has_children: true
has_toc: false
---

# Class `device_ptr_memory_resource`

Memory resource adaptor that turns any memory resource that returns a fancy with the same tag as <code><a href="/api/classes/classdevice__ptr.html">device&#95;ptr</a></code>, and adapts it to a resource that returns a <code><a href="/api/classes/classdevice__ptr.html">device&#95;ptr</a></code>. 

**Inherits From**:
`thrust::mr::memory_resource< device_ptr< void > >`

<code class="doxybook">
<span>#include <thrust/device_allocator.h></span><br>
<span>template &lt;typename Upstream&gt;</span>
<span>class device&#95;ptr&#95;memory&#95;resource {</span>
<span>public:</span><span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classdevice__ptr__memory__resource.html#function-device_ptr_memory_resource">device&#95;ptr&#95;memory&#95;resource</a></b>();</span>
<br>
<span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classdevice__ptr__memory__resource.html#function-device_ptr_memory_resource">device&#95;ptr&#95;memory&#95;resource</a></b>(Upstream * upstream);</span>
<br>
<span>&nbsp;&nbsp;virtual __host__ pointer </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classdevice__ptr__memory__resource.html#function-do_allocate">do&#95;allocate</a></b>(std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t)) override;</span>
<br>
<span>&nbsp;&nbsp;virtual __host__ void </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classdevice__ptr__memory__resource.html#function-do_deallocate">do&#95;deallocate</a></b>(pointer p,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment) override;</span>
<span>};</span>
</code>

## Member Functions

<h3 id="function-device_ptr_memory_resource">
Function <code>device&#95;ptr&#95;memory&#95;resource::&gt;::device&#95;ptr&#95;memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>__host__ </span><span><b>device_ptr_memory_resource</b>();</span></code>
Initialize the adaptor with the global instance of the upstream resource. Obtains the global instance by calling <code>get&#95;global&#95;resource</code>. 

<h3 id="function-device_ptr_memory_resource">
Function <code>device&#95;ptr&#95;memory&#95;resource::&gt;::device&#95;ptr&#95;memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>__host__ </span><span><b>device_ptr_memory_resource</b>(Upstream * upstream);</span></code>
Initialize the adaptor with an upstream resource.

**Function Parameters**:
**`upstream`**: the upstream memory resource to adapt. 

<h3 id="function-do_allocate">
Function <code>device&#95;ptr&#95;memory&#95;resource::&gt;::do&#95;allocate</code>
</h3>

<code class="doxybook">
<span>virtual __host__ pointer </span><span><b>do_allocate</b>(std::size_t bytes,</span>
<span>&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t)) override;</span></code>
<h3 id="function-do_deallocate">
Function <code>device&#95;ptr&#95;memory&#95;resource::&gt;::do&#95;deallocate</code>
</h3>

<code class="doxybook">
<span>virtual __host__ void </span><span><b>do_deallocate</b>(pointer p,</span>
<span>&nbsp;&nbsp;std::size_t bytes,</span>
<span>&nbsp;&nbsp;std::size_t alignment) override;</span></code>

