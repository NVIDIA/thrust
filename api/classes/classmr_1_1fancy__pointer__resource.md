---
title: mr::fancy_pointer_resource
nav_exclude: true
has_children: true
has_toc: false
---

# Class `mr::fancy_pointer_resource`

**Inherits From**:
* `mr::memory_resource< Pointer >`
* `mr::validator< Upstream >`

<code class="doxybook">
<span>template &lt;typename Upstream,</span>
<span>&nbsp;&nbsp;typename Pointer&gt;</span>
<span>class mr::fancy&#95;pointer&#95;resource {</span>
<span>public:</span><span class="doxybook-comment">/* Inherited from <code><b><a href="/api/classes/classmr_1_1memory__resource.html">mr::memory&#95;resource&lt; Pointer &gt;</a></b></code> */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="/api/classes/classmr_1_1memory__resource.html#typedef-pointer">pointer</a></b>;</span>
<br>
<span>&nbsp;&nbsp;<b><a href="/api/classes/classmr_1_1fancy__pointer__resource.html#function-fancy_pointer_resource">fancy&#95;pointer&#95;resource</a></b>();</span>
<br>
<span>&nbsp;&nbsp;<b><a href="/api/classes/classmr_1_1fancy__pointer__resource.html#function-fancy_pointer_resource">fancy&#95;pointer&#95;resource</a></b>(Upstream * upstream);</span>
<br>
<span>&nbsp;&nbsp;virtual Pointer </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classmr_1_1fancy__pointer__resource.html#function-do_allocate">do&#95;allocate</a></b>(std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t)) override;</span>
<br>
<span>&nbsp;&nbsp;virtual void </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classmr_1_1fancy__pointer__resource.html#function-do_deallocate">do&#95;deallocate</a></b>(Pointer p,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment) override;</span>
<span class="doxybook-comment">/* Inherited from <code><b><a href="/api/classes/classmr_1_1memory__resource.html">mr::memory&#95;resource&lt; Pointer &gt;</a></b></code> */</span><br>
<span>&nbsp;&nbsp;virtual </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classmr_1_1memory__resource.html#function-~memory_resource">~memory&#95;resource</a></b>() = default;</span>
<span class="doxybook-comment">/* Inherited from <code><b><a href="/api/classes/classmr_1_1memory__resource.html">mr::memory&#95;resource&lt; Pointer &gt;</a></b></code> */</span><br>
<span>&nbsp;&nbsp;<a href="/api/classes/classmr_1_1memory__resource.html#typedef-pointer">pointer</a> </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classmr_1_1memory__resource.html#function-allocate">allocate</a></b>(std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t));</span>
<span class="doxybook-comment">/* Inherited from <code><b><a href="/api/classes/classmr_1_1memory__resource.html">mr::memory&#95;resource&lt; Pointer &gt;</a></b></code> */</span><br>
<span>&nbsp;&nbsp;void </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classmr_1_1memory__resource.html#function-deallocate">deallocate</a></b>(<a href="/api/classes/classmr_1_1memory__resource.html#typedef-pointer">pointer</a> p,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t));</span>
<span class="doxybook-comment">/* Inherited from <code><b><a href="/api/classes/classmr_1_1memory__resource.html">mr::memory&#95;resource&lt; Pointer &gt;</a></b></code> */</span><br>
<span>&nbsp;&nbsp;__host__ __device__ bool </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classmr_1_1memory__resource.html#function-is_equal">is&#95;equal</a></b>(const <a href="/api/classes/classmr_1_1memory__resource.html">memory_resource</a> & other) const;</span>
<span class="doxybook-comment">/* Inherited from <code><b><a href="/api/classes/classmr_1_1memory__resource.html">mr::memory&#95;resource&lt; Pointer &gt;</a></b></code> */</span><br>
<span>&nbsp;&nbsp;virtual __host__ virtual __device__ bool </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classmr_1_1memory__resource.html#function-do_is_equal">do&#95;is&#95;equal</a></b>(const <a href="/api/classes/classmr_1_1memory__resource.html">memory_resource</a> & other) const;</span>
<span>};</span>
</code>

## Member Functions

<h3 id="function-fancy_pointer_resource">
Function <code>mr::fancy&#95;pointer&#95;resource::&gt;::fancy&#95;pointer&#95;resource</code>
</h3>

<code class="doxybook">
<span><b>fancy_pointer_resource</b>();</span></code>
<h3 id="function-fancy_pointer_resource">
Function <code>mr::fancy&#95;pointer&#95;resource::&gt;::fancy&#95;pointer&#95;resource</code>
</h3>

<code class="doxybook">
<span><b>fancy_pointer_resource</b>(Upstream * upstream);</span></code>
<h3 id="function-do_allocate">
Function <code>mr::fancy&#95;pointer&#95;resource::&gt;::do&#95;allocate</code>
</h3>

<code class="doxybook">
<span>virtual Pointer </span><span><b>do_allocate</b>(std::size_t bytes,</span>
<span>&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t)) override;</span></code>
<h3 id="function-do_deallocate">
Function <code>mr::fancy&#95;pointer&#95;resource::&gt;::do&#95;deallocate</code>
</h3>

<code class="doxybook">
<span>virtual void </span><span><b>do_deallocate</b>(Pointer p,</span>
<span>&nbsp;&nbsp;std::size_t bytes,</span>
<span>&nbsp;&nbsp;std::size_t alignment) override;</span></code>

