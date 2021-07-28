---
title: mr::memory_resource
parent: Memory Resources
grand_parent: Memory Management
nav_exclude: true
has_children: true
has_toc: false
---

# Class `mr::memory_resource`

<code><a href="/api/classes/classmr_1_1memory__resource.html">memory&#95;resource</a></code> is the base class for all other memory resources.

**Template Parameters**:
**`Pointer`**: the pointer type that is allocated and deallocated by the memory resource derived from this base class. If this is <code>void &#42;</code>, this class derives from <code>std::pmr::memory&#95;resource</code>. 

**Inherited By**:
* [`mr::fancy_pointer_resource< Upstream, Pointer >`](/api/classes/classmr_1_1fancy__pointer__resource.html)
* [`mr::polymorphic_adaptor_resource< Pointer >`](/api/classes/classmr_1_1polymorphic__adaptor__resource.html)

<code class="doxybook">
<span>#include <thrust/mr/memory_resource.h></span><br>
<span>template &lt;typename Pointer = void &#42;&gt;</span>
<span>class mr::memory&#95;resource {</span>
<span>public:</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="/api/classes/classmr_1_1memory__resource.html#typedef-pointer">pointer</a></b>;</span>
<br>
<span>&nbsp;&nbsp;virtual </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classmr_1_1memory__resource.html#function-~memory_resource">~memory&#95;resource</a></b>() = default;</span>
<br>
<span>&nbsp;&nbsp;<a href="/api/classes/classmr_1_1memory__resource.html#typedef-pointer">pointer</a> </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classmr_1_1memory__resource.html#function-allocate">allocate</a></b>(std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t));</span>
<br>
<span>&nbsp;&nbsp;void </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classmr_1_1memory__resource.html#function-deallocate">deallocate</a></b>(<a href="/api/classes/classmr_1_1memory__resource.html#typedef-pointer">pointer</a> p,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t));</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ bool </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classmr_1_1memory__resource.html#function-is_equal">is&#95;equal</a></b>(const <a href="/api/classes/classmr_1_1memory__resource.html">memory_resource</a> & other) const;</span>
<br>
<span>&nbsp;&nbsp;virtual <a href="/api/classes/classmr_1_1memory__resource.html#typedef-pointer">pointer</a> </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classmr_1_1memory__resource.html#function-do_allocate">do&#95;allocate</a></b>(std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment) = 0;</span>
<br>
<span>&nbsp;&nbsp;virtual void </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classmr_1_1memory__resource.html#function-do_deallocate">do&#95;deallocate</a></b>(<a href="/api/classes/classmr_1_1memory__resource.html#typedef-pointer">pointer</a> p,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t bytes,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::size_t alignment) = 0;</span>
<br>
<span>&nbsp;&nbsp;virtual __host__ virtual __device__ bool </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classmr_1_1memory__resource.html#function-do_is_equal">do&#95;is&#95;equal</a></b>(const <a href="/api/classes/classmr_1_1memory__resource.html">memory_resource</a> & other) const;</span>
<span>};</span>
</code>

## Member Types

<h3 id="typedef-pointer">
Typedef <code>mr::memory&#95;resource::pointer</code>
</h3>

<code class="doxybook">
<span>typedef Pointer<b>pointer</b>;</span></code>
Alias for the template parameter. 


## Member Functions

<h3 id="function-~memory_resource">
Function <code>mr::memory&#95;resource::&gt;::~memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>virtual </span><span><b>~memory_resource</b>() = default;</span></code>
Virtual destructor, defaulted when possible. 

<h3 id="function-allocate">
Function <code>mr::memory&#95;resource::&gt;::allocate</code>
</h3>

<code class="doxybook">
<span><a href="/api/classes/classmr_1_1memory__resource.html#typedef-pointer">pointer</a> </span><span><b>allocate</b>(std::size_t bytes,</span>
<span>&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t));</span></code>
Allocates memory of size at least <code>bytes</code> and alignment at least <code>alignment</code>.

**Function Parameters**:
* **`bytes`** size, in bytes, that is requested from this allocation 
* **`alignment`** alignment that is requested from this allocation 

**Returns**:
A pointer to void to the newly allocated memory. 

**Exceptions**:
**`thrust::bad_alloc`**: when no memory with requested size and alignment can be allocated. 

<h3 id="function-deallocate">
Function <code>mr::memory&#95;resource::&gt;::deallocate</code>
</h3>

<code class="doxybook">
<span>void </span><span><b>deallocate</b>(<a href="/api/classes/classmr_1_1memory__resource.html#typedef-pointer">pointer</a> p,</span>
<span>&nbsp;&nbsp;std::size_t bytes,</span>
<span>&nbsp;&nbsp;std::size_t alignment = alignof(std::max&#95;align&#95;t));</span></code>
Deallocates memory pointed to by <code>p</code>.

**Function Parameters**:
* **`p`** pointer to be deallocated 
* **`bytes`** the size of the allocation. This must be equivalent to the value of <code>bytes</code> that was passed to the allocation function that returned <code>p</code>. 
* **`alignment`** the alignment of the allocation. This must be equivalent to the value of <code>alignment</code> that was passed to the allocation function that returned <code>p</code>. 

<h3 id="function-is_equal">
Function <code>mr::memory&#95;resource::&gt;::is&#95;equal</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ bool </span><span><b>is_equal</b>(const <a href="/api/classes/classmr_1_1memory__resource.html">memory_resource</a> & other) const;</span></code>
Compares this resource to the other one. The default implementation uses identity comparison, which is often the right thing to do and doesn't require RTTI involvement.

**Function Parameters**:
**`other`**: the other resource to compare this resource to 

**Returns**:
whether the two resources are equivalent. 

<h3 id="function-do_allocate">
Function <code>mr::memory&#95;resource::&gt;::do&#95;allocate</code>
</h3>

<code class="doxybook">
<span>virtual <a href="/api/classes/classmr_1_1memory__resource.html#typedef-pointer">pointer</a> </span><span><b>do_allocate</b>(std::size_t bytes,</span>
<span>&nbsp;&nbsp;std::size_t alignment) = 0;</span></code>
Allocates memory of size at least <code>bytes</code> and alignment at least <code>alignment</code>.

**Function Parameters**:
* **`bytes`** size, in bytes, that is requested from this allocation 
* **`alignment`** alignment that is requested from this allocation 

**Returns**:
A pointer to void to the newly allocated memory. 

**Exceptions**:
**`thrust::bad_alloc`**: when no memory with requested size and alignment can be allocated. 

**Implemented By**:
* [`do_allocate`](/api/classes/classmr_1_1new__delete__resource.html#function-do_allocate)
* [`do_allocate`](/api/classes/structmr_1_1synchronized__pool__resource.html#function-do_allocate)
* [`do_allocate`](/api/classes/structmr_1_1disjoint__synchronized__pool__resource.html#function-do_allocate)
* [`do_allocate`](/api/classes/classmr_1_1unsynchronized__pool__resource.html#function-do_allocate)
* [`do_allocate`](/api/classes/classmr_1_1disjoint__unsynchronized__pool__resource.html#function-do_allocate)

<h3 id="function-do_deallocate">
Function <code>mr::memory&#95;resource::&gt;::do&#95;deallocate</code>
</h3>

<code class="doxybook">
<span>virtual void </span><span><b>do_deallocate</b>(<a href="/api/classes/classmr_1_1memory__resource.html#typedef-pointer">pointer</a> p,</span>
<span>&nbsp;&nbsp;std::size_t bytes,</span>
<span>&nbsp;&nbsp;std::size_t alignment) = 0;</span></code>
Deallocates memory pointed to by <code>p</code>.

**Function Parameters**:
* **`p`** pointer to be deallocated 
* **`bytes`** the size of the allocation. This must be equivalent to the value of <code>bytes</code> that was passed to the allocation function that returned <code>p</code>. 
* **`alignment`** the size of the allocation. This must be equivalent to the value of <code>alignment</code> that was passed to the allocation function that returned <code>p</code>. 

**Implemented By**:
* [`do_deallocate`](/api/classes/classmr_1_1new__delete__resource.html#function-do_deallocate)
* [`do_deallocate`](/api/classes/structmr_1_1synchronized__pool__resource.html#function-do_deallocate)
* [`do_deallocate`](/api/classes/structmr_1_1disjoint__synchronized__pool__resource.html#function-do_deallocate)
* [`do_deallocate`](/api/classes/classmr_1_1unsynchronized__pool__resource.html#function-do_deallocate)
* [`do_deallocate`](/api/classes/classmr_1_1disjoint__unsynchronized__pool__resource.html#function-do_deallocate)

<h3 id="function-do_is_equal">
Function <code>mr::memory&#95;resource::&gt;::do&#95;is&#95;equal</code>
</h3>

<code class="doxybook">
<span>virtual __host__ virtual __device__ bool </span><span><b>do_is_equal</b>(const <a href="/api/classes/classmr_1_1memory__resource.html">memory_resource</a> & other) const;</span></code>
Compares this resource to the other one. The default implementation uses identity comparison, which is often the right thing to do and doesn't require RTTI involvement.

**Function Parameters**:
**`other`**: the other resource to compare this resource to 

**Returns**:
whether the two resources are equivalent. 


