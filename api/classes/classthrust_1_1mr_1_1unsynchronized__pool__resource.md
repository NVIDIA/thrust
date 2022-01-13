---
title: thrust::mr::unsynchronized_pool_resource
parent: Memory Resources
grand_parent: Memory Management
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::mr::unsynchronized_pool_resource`

A memory resource adaptor allowing for pooling and caching allocations from <code>Upstream</code>, using memory allocated from it for both blocks then allocated to the user and for internal bookkeeping of the cached memory.

On a typical memory resource, calls to <code>allocate</code> and <code>deallocate</code> actually allocate and deallocate memory. Pooling memory resources only allocate and deallocate memory from an external resource (the upstream memory resource) when there's no suitable memory currently cached; otherwise, they use memory they have acquired beforehand, to make memory allocation faster and more efficient.

The non-disjoint version of the pool resource uses a single upstream memory resource. Every allocation is larger than strictly necessary to fulfill the end-user's request, because it needs to account for the memory overhead of tracking the memory blocks and chunks inside those same memory regions. Nevertheless, this version should be more memory-efficient than the <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1disjoint__unsynchronized__pool__resource.html">disjoint&#95;unsynchronized&#95;pool&#95;resource</a></code>, because it doesn't need to allocate additional blocks of memory from a separate resource, which in turn would necessitate the bookkeeping overhead in the upstream resource.

This version requires that memory allocated from Upstream is accessible from device. It supports smart references, meaning that the non-managed CUDA resource, returning a device-tagged pointer, will work, but will be much less efficient than the disjoint version, which wouldn't need to touch device memory at all, and therefore wouldn't need to transfer it back and forth between the host and the device whenever an allocation or a deallocation happens.

**Template Parameters**:
**`Upstream`**: the type of memory resources that will be used for allocating memory blocks 

**Inherits From**:
* `thrust::mr::memory_resource< Upstream::pointer >`
* `thrust::mr::validator< Upstream >`

<code class="doxybook">
<span>#include <thrust/mr/pool.h></span><br>
<span>template &lt;typename Upstream&gt;</span>
<span>class thrust::mr::unsynchronized&#95;pool&#95;resource {</span>
<span>public:</span><span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">thrust::mr::memory&#95;resource&lt; Upstream::pointer &gt;</a></b></code> */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#typedef-pointer">pointer</a></b>;</span>
<br>
<span>&nbsp;&nbsp;static <a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1pool__options.html">pool_options</a> </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1unsynchronized__pool__resource.html#function-get-default-options">get&#95;default&#95;options</a></b>();</span>
<br>
<span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1unsynchronized__pool__resource.html#function-unsynchronized-pool-resource">unsynchronized&#95;pool&#95;resource</a></b>(Upstream * upstream,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1pool__options.html">pool_options</a> options = get&#95;default&#95;options());</span>
<br>
<span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1unsynchronized__pool__resource.html#function-unsynchronized-pool-resource">unsynchronized&#95;pool&#95;resource</a></b>(<a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1pool__options.html">pool_options</a> options = get&#95;default&#95;options());</span>
<br>
<span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1unsynchronized__pool__resource.html#function-~unsynchronized-pool-resource">~unsynchronized&#95;pool&#95;resource</a></b>();</span>
<br>
<span>&nbsp;&nbsp;void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1unsynchronized__pool__resource.html#function-release">release</a></b>();</span>
<br>
<span>&nbsp;&nbsp;virtual void_ptr </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1unsynchronized__pool__resource.html#function-do-allocate">do&#95;allocate</a></b>(std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t)) override;</span>
<br>
<span>&nbsp;&nbsp;virtual void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1unsynchronized__pool__resource.html#function-do-deallocate">do&#95;deallocate</a></b>(void_ptr p,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t)) override;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">thrust::mr::memory&#95;resource&lt; Upstream::pointer &gt;</a></b></code> */</span><span>&nbsp;&nbsp;virtual </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#function-~memory-resource">~memory&#95;resource</a></b>() = default;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">thrust::mr::memory&#95;resource&lt; Upstream::pointer &gt;</a></b></code> */</span><span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#typedef-pointer">pointer</a> </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#function-allocate">allocate</a></b>(std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t));</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">thrust::mr::memory&#95;resource&lt; Upstream::pointer &gt;</a></b></code> */</span><span>&nbsp;&nbsp;void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#function-deallocate">deallocate</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#typedef-pointer">pointer</a> p,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t));</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">thrust::mr::memory&#95;resource&lt; Upstream::pointer &gt;</a></b></code> */</span><span>&nbsp;&nbsp;__host__ __device__ bool </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#function-is-equal">is&#95;equal</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">memory_resource</a> & other) const;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">thrust::mr::memory&#95;resource&lt; Upstream::pointer &gt;</a></b></code> */</span><span>&nbsp;&nbsp;virtual __host__ virtual __device__ bool </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#function-do-is-equal">do&#95;is&#95;equal</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">memory_resource</a> & other) const;</span>
<span>};</span>
</code>

## Member Functions

<h3 id="function-get-default-options">
Function <code>thrust::mr::unsynchronized&#95;pool&#95;resource::get&#95;default&#95;options</code>
</h3>

<code class="doxybook">
<span>static <a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1pool__options.html">pool_options</a> </span><span><b>get_default_options</b>();</span></code>
Get the default options for a pool. These are meant to be a sensible set of values for many use cases, and as such, may be tuned in the future. This function is exposed so that creating a set of options that are just a slight departure from the defaults is easy. 

<h3 id="function-unsynchronized-pool-resource">
Function <code>thrust::mr::unsynchronized&#95;pool&#95;resource::unsynchronized&#95;pool&#95;resource</code>
</h3>

<code class="doxybook">
<span><b>unsynchronized_pool_resource</b>(Upstream * upstream,</span>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1pool__options.html">pool_options</a> options = get&#95;default&#95;options());</span></code>
Constructor.

**Function Parameters**:
* **`upstream`** the upstream memory resource for allocations 
* **`options`** pool options to use 

<h3 id="function-unsynchronized-pool-resource">
Function <code>thrust::mr::unsynchronized&#95;pool&#95;resource::unsynchronized&#95;pool&#95;resource</code>
</h3>

<code class="doxybook">
<span><b>unsynchronized_pool_resource</b>(<a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1pool__options.html">pool_options</a> options = get&#95;default&#95;options());</span></code>
Constructor. The upstream resource is obtained by calling <code>get&#95;global&#95;resource&lt;Upstream&gt;</code>.

**Function Parameters**:
**`options`**: pool options to use 

<h3 id="function-~unsynchronized-pool-resource">
Function <code>thrust::mr::unsynchronized&#95;pool&#95;resource::~unsynchronized&#95;pool&#95;resource</code>
</h3>

<code class="doxybook">
<span><b>~unsynchronized_pool_resource</b>();</span></code>
Destructor. Releases all held memory to upstream. 

<h3 id="function-release">
Function <code>thrust::mr::unsynchronized&#95;pool&#95;resource::release</code>
</h3>

<code class="doxybook">
<span>void </span><span><b>release</b>();</span></code>
Releases all held memory to upstream. 

<h3 id="function-do-allocate">
Function <code>thrust::mr::unsynchronized&#95;pool&#95;resource::do&#95;allocate</code>
</h3>

<code class="doxybook">
<span>virtual void_ptr </span><span><b>do_allocate</b>(std::size_t bytes,</span>
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
Function <code>thrust::mr::unsynchronized&#95;pool&#95;resource::do&#95;deallocate</code>
</h3>

<code class="doxybook">
<span>virtual void </span><span><b>do_deallocate</b>(void_ptr p,</span>
<span>&nbsp;&nbsp;std::size_t bytes,</span>
<span>&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t)) override;</span></code>
Deallocates memory pointed to by <code>p</code>.

**Function Parameters**:
* **`p`** pointer to be deallocated 
* **`bytes`** the size of the allocation. This must be equivalent to the value of <code>bytes</code> that was passed to the allocation function that returned <code>p</code>. 
* **`alignment`** the size of the allocation. This must be equivalent to the value of <code>alignment</code> that was passed to the allocation function that returned <code>p</code>. 

**Implements**: [`do_deallocate`]({{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#function-do-deallocate)


