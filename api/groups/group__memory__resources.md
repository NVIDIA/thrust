---
title: Memory Resources
parent: Memory Management
grand_parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Memory Resources

<code class="doxybook">
<span>template &lt;typename Upstream,</span>
<span>&nbsp;&nbsp;typename Bookkeeper&gt;</span>
<span>class <b><a href="/api/classes/classmr_1_1disjoint__unsynchronized__pool__resource.html">mr::disjoint&#95;unsynchronized&#95;pool&#95;resource</a></b>;</span>
<br>
<span>template &lt;typename Upstream,</span>
<span>&nbsp;&nbsp;typename Bookkeeper&gt;</span>
<span>struct <b><a href="/api/classes/structmr_1_1disjoint__synchronized__pool__resource.html">mr::disjoint&#95;synchronized&#95;pool&#95;resource</a></b>;</span>
<br>
<span>template &lt;typename Pointer = void &#42;&gt;</span>
<span>class <b><a href="/api/classes/classmr_1_1memory__resource.html">mr::memory&#95;resource</a></b>;</span>
<br>
<span>class <b><a href="/api/classes/classmr_1_1memory__resource_3_01void_01_5_01_4.html">mr::memory&#95;resource&lt; void &#42; &gt;</a></b>;</span>
<br>
<span>class <b><a href="/api/classes/classmr_1_1new__delete__resource.html">mr::new&#95;delete&#95;resource</a></b>;</span>
<br>
<span>template &lt;typename Upstream&gt;</span>
<span>class <b><a href="/api/classes/classmr_1_1unsynchronized__pool__resource.html">mr::unsynchronized&#95;pool&#95;resource</a></b>;</span>
<br>
<span>struct <b><a href="/api/classes/structmr_1_1pool__options.html">mr::pool&#95;options</a></b>;</span>
<br>
<span>template &lt;typename Upstream&gt;</span>
<span>struct <b><a href="/api/classes/structmr_1_1synchronized__pool__resource.html">mr::synchronized&#95;pool&#95;resource</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="/api/groups/group__memory__resources.html#typedef-memory_resource">memory&#95;resource</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="/api/groups/group__memory__resources.html#typedef-universal_memory_resource">universal&#95;memory&#95;resource</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="/api/groups/group__memory__resources.html#typedef-universal_host_pinned_memory_resource">universal&#95;host&#95;pinned&#95;memory&#95;resource</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="/api/groups/group__memory__resources.html#typedef-memory_resource">memory&#95;resource</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="/api/groups/group__memory__resources.html#typedef-universal_memory_resource">universal&#95;memory&#95;resource</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="/api/groups/group__memory__resources.html#typedef-universal_host_pinned_memory_resource">universal&#95;host&#95;pinned&#95;memory&#95;resource</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="/api/groups/group__memory__resources.html#typedef-memory_resource">memory&#95;resource</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="/api/groups/group__memory__resources.html#typedef-universal_memory_resource">universal&#95;memory&#95;resource</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="/api/groups/group__memory__resources.html#typedef-universal_host_pinned_memory_resource">universal&#95;host&#95;pinned&#95;memory&#95;resource</a></b>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>using <b><a href="/api/groups/group__memory__resources.html#using-universal_ptr">universal&#95;ptr</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;typename Upstream,</span>
<span>&nbsp;&nbsp;typename Bookkeeper&gt;</span>
<span>__host__ thrust::mr::disjoint_unsynchronized_pool_resource< Upstream, Bookkeeper > & </span><span><b><a href="/api/groups/group__memory__resources.html#function-tls_disjoint_pool">mr::tls&#95;disjoint&#95;pool</a></b>(Upstream * upstream = NULL,</span>
<span>&nbsp;&nbsp;Bookkeeper * bookkeeper = NULL);</span>
<br>
<span>template &lt;typename Pointer&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="/api/groups/group__memory__resources.html#function-operator==">mr::operator==</a></b>(const memory_resource< Pointer > & lhs,</span>
<span>&nbsp;&nbsp;const memory_resource< Pointer > & rhs);</span>
<br>
<span>template &lt;typename Pointer&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="/api/groups/group__memory__resources.html#function-operator!=">mr::operator!=</a></b>(const memory_resource< Pointer > & lhs,</span>
<span>&nbsp;&nbsp;const memory_resource< Pointer > & rhs);</span>
<br>
<span>template &lt;typename MR&gt;</span>
<span>__host__ MR * </span><span><b><a href="/api/groups/group__memory__resources.html#function-get_global_resource">mr::get&#95;global&#95;resource</a></b>();</span>
<br>
<span>template &lt;typename Upstream,</span>
<span>&nbsp;&nbsp;typename Bookkeeper&gt;</span>
<span>__host__ thrust::mr::unsynchronized_pool_resource< Upstream > & </span><span><b><a href="/api/groups/group__memory__resources.html#function-tls_pool">mr::tls&#95;pool</a></b>(Upstream * upstream = NULL);</span>
</code>

## Member Classes

<h3 id="class-mr::disjoint_unsynchronized_pool_resource">
<a href="/api/classes/classmr_1_1disjoint__unsynchronized__pool__resource.html">Class <code>mr::disjoint&#95;unsynchronized&#95;pool&#95;resource</code>
</a>
</h3>

**Inherits From**:
* `mr::memory_resource< Upstream::pointer >`
* `mr::validator2< Upstream, Bookkeeper >`

<h3 id="struct-mr::disjoint_synchronized_pool_resource">
<a href="/api/classes/structmr_1_1disjoint__synchronized__pool__resource.html">Struct <code>mr::disjoint&#95;synchronized&#95;pool&#95;resource</code>
</a>
</h3>

**Inherits From**:
[`mr::memory_resource< Upstream::pointer >`](/api/classes/classmr_1_1memory__resource.html)

<h3 id="class-mr::memory_resource">
<a href="/api/classes/classmr_1_1memory__resource.html">Class <code>mr::memory&#95;resource</code>
</a>
</h3>

**Inherited By**:
* [`mr::fancy_pointer_resource< Upstream, Pointer >`](/api/classes/classmr_1_1fancy__pointer__resource.html)
* [`mr::polymorphic_adaptor_resource< Pointer >`](/api/classes/classmr_1_1polymorphic__adaptor__resource.html)

<h3 id="class-mr::memory_resource< void * >">
<a href="/api/classes/classmr_1_1memory__resource_3_01void_01_5_01_4.html">Class <code>mr::memory&#95;resource&lt; void &#42; &gt;</code>
</a>
</h3>

<h3 id="class-mr::new_delete_resource">
<a href="/api/classes/classmr_1_1new__delete__resource.html">Class <code>mr::new&#95;delete&#95;resource</code>
</a>
</h3>

**Inherits From**:
[`mr::memory_resource<>`](/api/classes/classmr_1_1memory__resource.html)

<h3 id="class-mr::unsynchronized_pool_resource">
<a href="/api/classes/classmr_1_1unsynchronized__pool__resource.html">Class <code>mr::unsynchronized&#95;pool&#95;resource</code>
</a>
</h3>

**Inherits From**:
* `mr::memory_resource< Upstream::pointer >`
* `mr::validator< Upstream >`

<h3 id="struct-mr::pool_options">
<a href="/api/classes/structmr_1_1pool__options.html">Struct <code>mr::pool&#95;options</code>
</a>
</h3>

<h3 id="struct-mr::synchronized_pool_resource">
<a href="/api/classes/structmr_1_1synchronized__pool__resource.html">Struct <code>mr::synchronized&#95;pool&#95;resource</code>
</a>
</h3>

**Inherits From**:
[`mr::memory_resource< Upstream::pointer >`](/api/classes/classmr_1_1memory__resource.html)


## Types

<h3 id="typedef-memory_resource">
Typedef <code>memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>typedef detail::native_resource<b>memory_resource</b>;</span></code>
The memory resource for the Standard C++ system. Uses <code><a href="/api/classes/classmr_1_1new__delete__resource.html">mr::new&#95;delete&#95;resource</a></code> and tags it with <code>cpp::pointer</code>. 

<h3 id="typedef-universal_memory_resource">
Typedef <code>universal&#95;memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>typedef detail::universal_native_resource<b>universal_memory_resource</b>;</span></code>
The unified memory resource for the Standard C++ system. Uses <code><a href="/api/classes/classmr_1_1new__delete__resource.html">mr::new&#95;delete&#95;resource</a></code> and tags it with <code>cpp::universal&#95;pointer</code>. 

<h3 id="typedef-universal_host_pinned_memory_resource">
Typedef <code>universal&#95;host&#95;pinned&#95;memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>typedef detail::native_resource<b>universal_host_pinned_memory_resource</b>;</span></code>
An alias for <code>cpp::universal&#95;memory&#95;resource</code>. 

<h3 id="typedef-memory_resource">
Typedef <code>memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>typedef detail::native_resource<b>memory_resource</b>;</span></code>
The memory resource for the OpenMP system. Uses <code><a href="/api/classes/classmr_1_1new__delete__resource.html">mr::new&#95;delete&#95;resource</a></code> and tags it with <code>omp::pointer</code>. 

<h3 id="typedef-universal_memory_resource">
Typedef <code>universal&#95;memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>typedef detail::universal_native_resource<b>universal_memory_resource</b>;</span></code>
The unified memory resource for the OpenMP system. Uses <code><a href="/api/classes/classmr_1_1new__delete__resource.html">mr::new&#95;delete&#95;resource</a></code> and tags it with <code>omp::universal&#95;pointer</code>. 

<h3 id="typedef-universal_host_pinned_memory_resource">
Typedef <code>universal&#95;host&#95;pinned&#95;memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>typedef detail::native_resource<b>universal_host_pinned_memory_resource</b>;</span></code>
An alias for <code>omp::universal&#95;memory&#95;resource</code>. 

<h3 id="typedef-memory_resource">
Typedef <code>memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>typedef detail::native_resource<b>memory_resource</b>;</span></code>
The memory resource for the TBB system. Uses <code><a href="/api/classes/classmr_1_1new__delete__resource.html">mr::new&#95;delete&#95;resource</a></code> and tags it with <code>tbb::pointer</code>. 

<h3 id="typedef-universal_memory_resource">
Typedef <code>universal&#95;memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>typedef detail::universal_native_resource<b>universal_memory_resource</b>;</span></code>
The unified memory resource for the TBB system. Uses <code><a href="/api/classes/classmr_1_1new__delete__resource.html">mr::new&#95;delete&#95;resource</a></code> and tags it with <code>tbb::universal&#95;pointer</code>. 

<h3 id="typedef-universal_host_pinned_memory_resource">
Typedef <code>universal&#95;host&#95;pinned&#95;memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>typedef detail::native_resource<b>universal_host_pinned_memory_resource</b>;</span></code>
An alias for <code>tbb::universal&#95;memory&#95;resource</code>. 

<h3 id="using-universal_ptr">
Type Alias <code>universal&#95;ptr</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>universal_ptr</b> = thrust::system::&#95;&#95;THRUST&#95;DEVICE&#95;SYSTEM&#95;NAMESPACE::universal&#95;pointer&lt; T &gt;;</span></code>
<code>universal&#95;ptr</code> stores a pointer to an object allocated in memory accessible to both hosts and devices.

Algorithms dispatched with this type of pointer will be dispatched to either host or device, depending on which backend you are using. Explicit policies (<code>thrust::device</code>, etc) can be used to specify where an algorithm should be run.

<code>universal&#95;ptr</code> has pointer semantics: it may be dereferenced safely from both hosts and devices and may be manipulated with pointer arithmetic.

<code>universal&#95;ptr</code> can be created with <code>universal&#95;allocator</code> or by explicitly calling its constructor with a raw pointer.

The raw pointer encapsulated by a <code>universal&#95;ptr</code> may be obtained by either its <code>get</code> method or the <code>raw&#95;pointer&#95;cast</code> free function.

**Note**:
<code>universal&#95;ptr</code> is not a smart pointer; it is the programmer's responsibility to deallocate memory pointed to by <code>universal&#95;ptr</code>.

**See**:
* host_ptr For the documentation of the complete interface which is shared by <code><a href="/api/groups/group__memory__resources.html#using-universal_ptr">universal&#95;ptr</a></code>. 
* <a href="/api/groups/group__memory__management.html#function-raw_pointer_cast">raw_pointer_cast</a>


## Functions

<h3 id="function-tls_disjoint_pool">
Function <code>mr::tls&#95;disjoint&#95;pool</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Upstream,</span>
<span>&nbsp;&nbsp;typename Bookkeeper&gt;</span>
<span>__host__ thrust::mr::disjoint_unsynchronized_pool_resource< Upstream, Bookkeeper > & </span><span><b>tls_disjoint_pool</b>(Upstream * upstream = NULL,</span>
<span>&nbsp;&nbsp;Bookkeeper * bookkeeper = NULL);</span></code>
Potentially constructs, if not yet created, and then returns the address of a thread-local <code><a href="/api/classes/classmr_1_1disjoint__unsynchronized__pool__resource.html">disjoint&#95;unsynchronized&#95;pool&#95;resource</a></code>,

**Template Parameters**:
* **`Upstream`** the first template argument to the pool template 
* **`Bookkeeper`** the second template argument to the pool template 

**Function Parameters**:
* **`upstream`** the first argument to the constructor, if invoked 
* **`bookkeeper`** the second argument to the constructor, if invoked 

<h3 id="function-operator==">
Function <code>mr::operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Pointer&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const memory_resource< Pointer > & lhs,</span>
<span>&nbsp;&nbsp;const memory_resource< Pointer > & rhs);</span></code>
Compares the memory resources for equality, first by identity, then by <code>is&#95;equal</code>. 

<h3 id="function-operator!=">
Function <code>mr::operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Pointer&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const memory_resource< Pointer > & lhs,</span>
<span>&nbsp;&nbsp;const memory_resource< Pointer > & rhs);</span></code>
Compares the memory resources for inequality, first by identity, then by <code>is&#95;equal</code>. 

<h3 id="function-get_global_resource">
Function <code>mr::get&#95;global&#95;resource</code>
</h3>

<code class="doxybook">
<span>template &lt;typename MR&gt;</span>
<span>__host__ MR * </span><span><b>get_global_resource</b>();</span></code>
Returns a global instance of <code>MR</code>, created as a function local static variable.

**Template Parameters**:
**`MR`**: type of a memory resource to get an instance from. Must be <code>DefaultConstructible</code>. 

**Returns**:
a pointer to a global instance of <code>MR</code>. 

<h3 id="function-tls_pool">
Function <code>mr::tls&#95;pool</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Upstream,</span>
<span>&nbsp;&nbsp;typename Bookkeeper&gt;</span>
<span>__host__ thrust::mr::unsynchronized_pool_resource< Upstream > & </span><span><b>tls_pool</b>(Upstream * upstream = NULL);</span></code>
Potentially constructs, if not yet created, and then returns the address of a thread-local <code><a href="/api/classes/classmr_1_1unsynchronized__pool__resource.html">unsynchronized&#95;pool&#95;resource</a></code>,

**Template Parameters**:
**`Upstream`**: the template argument to the pool template 

**Function Parameters**:
**`upstream`**: the argument to the constructor, if invoked 


