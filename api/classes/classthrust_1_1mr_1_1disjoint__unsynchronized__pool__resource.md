---
title: thrust::mr::disjoint_unsynchronized_pool_resource
parent: Memory Resources
grand_parent: Memory Management
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::mr::disjoint_unsynchronized_pool_resource`

A memory resource adaptor allowing for pooling and caching allocations from <code>Upstream</code>, using <code>Bookkeeper</code> for management of that cached and pooled memory, allowing to cache portions of memory inaccessible from the host.

On a typical memory resource, calls to <code>allocate</code> and <code>deallocate</code> actually allocate and deallocate memory. Pooling memory resources only allocate and deallocate memory from an external resource (the upstream memory resource) when there's no suitable memory currently cached; otherwise, they use memory they have acquired beforehand, to make memory allocation faster and more efficient.

The disjoint version of the pool resources uses a separate upstream memory resource, <code>Bookkeeper</code>, to allocate memory necessary to manage the cached memory. There may be many reasons to do that; the canonical one is that <code>Upstream</code> allocates memory that is inaccessible to the code of the pool resource, which means that it cannot embed the necessary information in memory obtained from <code>Upstream</code>; for instance, <code>Upstream</code> can be a CUDA non-managed memory resource, or a CUDA managed memory resource whose memory we would prefer to not migrate back and forth between host and device when executing bookkeeping code.

This is not the only case where it makes sense to use a disjoint pool resource, though. In a multi-core environment it may be beneficial to avoid stealing cache lines from other cores by writing over bookkeeping information embedded in an allocated block of memory. In such a case, one can imagine wanting to use a disjoint pool where both the upstream and the bookkeeper are of the same type, to allocate memory consistently, but separately for those two purposes.

**Template Parameters**:
* **`Upstream`** the type of memory resources that will be used for allocating memory blocks to be handed off to the user 
* **`Bookkeeper`** the type of memory resources that will be used for allocating bookkeeping memory 

**Inherits From**:
* `thrust::mr::memory_resource< Upstream::pointer >`
* `thrust::mr::validator2< Upstream, Bookkeeper >`

<code class="doxybook">
<span>#include <thrust/mr/disjoint_pool.h></span><br>
<span>template &lt;typename Upstream,</span>
<span>&nbsp;&nbsp;typename Bookkeeper&gt;</span>
<span>class thrust::mr::disjoint&#95;unsynchronized&#95;pool&#95;resource {</span>
<span>public:</span><span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">thrust::mr::memory&#95;resource&lt; Upstream::pointer &gt;</a></b></code> */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html#typedef-pointer">pointer</a></b>;</span>
<br>
<span>&nbsp;&nbsp;static <a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1pool__options.html">pool_options</a> </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1disjoint__unsynchronized__pool__resource.html#function-get-default-options">get&#95;default&#95;options</a></b>();</span>
<br>
<span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1disjoint__unsynchronized__pool__resource.html#function-disjoint-unsynchronized-pool-resource">disjoint&#95;unsynchronized&#95;pool&#95;resource</a></b>(Upstream * upstream,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;Bookkeeper * bookkeeper,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1pool__options.html">pool_options</a> options = get&#95;default&#95;options());</span>
<br>
<span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1disjoint__unsynchronized__pool__resource.html#function-disjoint-unsynchronized-pool-resource">disjoint&#95;unsynchronized&#95;pool&#95;resource</a></b>(<a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1pool__options.html">pool_options</a> options = get&#95;default&#95;options());</span>
<br>
<span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1disjoint__unsynchronized__pool__resource.html#function-~disjoint-unsynchronized-pool-resource">~disjoint&#95;unsynchronized&#95;pool&#95;resource</a></b>();</span>
<br>
<span>&nbsp;&nbsp;void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1disjoint__unsynchronized__pool__resource.html#function-release">release</a></b>();</span>
<br>
<span>&nbsp;&nbsp;virtual void_ptr </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1disjoint__unsynchronized__pool__resource.html#function-do-allocate">do&#95;allocate</a></b>(std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t)) override;</span>
<br>
<span>&nbsp;&nbsp;virtual void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1disjoint__unsynchronized__pool__resource.html#function-do-deallocate">do&#95;deallocate</a></b>(void_ptr p,</span>
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
Function <code>thrust::mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::get&#95;default&#95;options</code>
</h3>

<code class="doxybook">
<span>static <a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1pool__options.html">pool_options</a> </span><span><b>get_default_options</b>();</span></code>
Get the default options for a disjoint pool. These are meant to be a sensible set of values for many use cases, and as such, may be tuned in the future. This function is exposed so that creating a set of options that are just a slight departure from the defaults is easy. 

<h3 id="function-disjoint-unsynchronized-pool-resource">
Function <code>thrust::mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::disjoint&#95;unsynchronized&#95;pool&#95;resource</code>
</h3>

<code class="doxybook">
<span><b>disjoint_unsynchronized_pool_resource</b>(Upstream * upstream,</span>
<span>&nbsp;&nbsp;Bookkeeper * bookkeeper,</span>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1pool__options.html">pool_options</a> options = get&#95;default&#95;options());</span></code>
Constructor.

**Function Parameters**:
* **`upstream`** the upstream memory resource for allocations 
* **`bookkeeper`** the upstream memory resource for bookkeeping 
* **`options`** pool options to use 

<h3 id="function-disjoint-unsynchronized-pool-resource">
Function <code>thrust::mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::disjoint&#95;unsynchronized&#95;pool&#95;resource</code>
</h3>

<code class="doxybook">
<span><b>disjoint_unsynchronized_pool_resource</b>(<a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1pool__options.html">pool_options</a> options = get&#95;default&#95;options());</span></code>
Constructor. Upstream and bookkeeping resources are obtained by calling <code>get&#95;global&#95;resource</code> for their types.

**Function Parameters**:
**`options`**: pool options to use 

<h3 id="function-~disjoint-unsynchronized-pool-resource">
Function <code>thrust::mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::~disjoint&#95;unsynchronized&#95;pool&#95;resource</code>
</h3>

<code class="doxybook">
<span><b>~disjoint_unsynchronized_pool_resource</b>();</span></code>
Destructor. Releases all held memory to upstream. 

<h3 id="function-release">
Function <code>thrust::mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::release</code>
</h3>

<code class="doxybook">
<span>void </span><span><b>release</b>();</span></code>
Releases all held memory to upstream. 

<h3 id="function-do-allocate">
Function <code>thrust::mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::do&#95;allocate</code>
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
Function <code>thrust::mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::do&#95;deallocate</code>
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


