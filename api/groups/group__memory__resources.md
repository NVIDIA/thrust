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
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1disjoint__unsynchronized__pool__resource.html">thrust::mr::disjoint&#95;unsynchronized&#95;pool&#95;resource</a></b>;</span>
<br>
<span>template &lt;typename Upstream,</span>
<span>&nbsp;&nbsp;typename Bookkeeper&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1disjoint__synchronized__pool__resource.html">thrust::mr::disjoint&#95;synchronized&#95;pool&#95;resource</a></b>;</span>
<br>
<span>template &lt;typename Pointer = void &#42;&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">thrust::mr::memory&#95;resource</a></b>;</span>
<br>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource_3_01void_01_5_01_4.html">thrust::mr::memory&#95;resource&lt; void &#42; &gt;</a></b>;</span>
<br>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1new__delete__resource.html">thrust::mr::new&#95;delete&#95;resource</a></b>;</span>
<br>
<span>template &lt;typename Upstream&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1unsynchronized__pool__resource.html">thrust::mr::unsynchronized&#95;pool&#95;resource</a></b>;</span>
<br>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1pool__options.html">thrust::mr::pool&#95;options</a></b>;</span>
<br>
<span>template &lt;typename Upstream&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1synchronized__pool__resource.html">thrust::mr::synchronized&#95;pool&#95;resource</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__memory__resources.html#typedef-memory-resource">thrust::system::cpp::memory&#95;resource</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__memory__resources.html#typedef-universal-memory-resource">thrust::system::cpp::universal&#95;memory&#95;resource</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__memory__resources.html#typedef-universal-host-pinned-memory-resource">thrust::system::cpp::universal&#95;host&#95;pinned&#95;memory&#95;resource</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__memory__resources.html#typedef-memory-resource">thrust::system::omp::memory&#95;resource</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__memory__resources.html#typedef-universal-memory-resource">thrust::system::omp::universal&#95;memory&#95;resource</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__memory__resources.html#typedef-universal-host-pinned-memory-resource">thrust::system::omp::universal&#95;host&#95;pinned&#95;memory&#95;resource</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__memory__resources.html#typedef-memory-resource">thrust::system::tbb::memory&#95;resource</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__memory__resources.html#typedef-universal-memory-resource">thrust::system::tbb::universal&#95;memory&#95;resource</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__memory__resources.html#typedef-universal-host-pinned-memory-resource">thrust::system::tbb::universal&#95;host&#95;pinned&#95;memory&#95;resource</a></b>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__memory__resources.html#using-universal-ptr">thrust::universal&#95;ptr</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;typename Upstream,</span>
<span>&nbsp;&nbsp;typename Bookkeeper&gt;</span>
<span>__host__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1disjoint__unsynchronized__pool__resource.html">thrust::mr::disjoint_unsynchronized_pool_resource</a>< Upstream, Bookkeeper > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__memory__resources.html#function-tls-disjoint-pool">thrust::mr::tls&#95;disjoint&#95;pool</a></b>(Upstream * upstream = NULL,</span>
<span>&nbsp;&nbsp;Bookkeeper * bookkeeper = NULL);</span>
<br>
<span>template &lt;typename Pointer&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__memory__resources.html#function-operator==">thrust::mr::operator==</a></b>(const memory_resource< Pointer > & lhs,</span>
<span>&nbsp;&nbsp;const memory_resource< Pointer > & rhs);</span>
<br>
<span>template &lt;typename Pointer&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="{{ site.baseurl }}/api/groups/group__memory__resources.html#function-operator!=">thrust::mr::operator!=</a></b>(const memory_resource< Pointer > & lhs,</span>
<span>&nbsp;&nbsp;const memory_resource< Pointer > & rhs);</span>
<br>
<span>template &lt;typename MR&gt;</span>
<span>__host__ MR * </span><span><b><a href="{{ site.baseurl }}/api/groups/group__memory__resources.html#function-get-global-resource">thrust::mr::get&#95;global&#95;resource</a></b>();</span>
<br>
<span>template &lt;typename Upstream,</span>
<span>&nbsp;&nbsp;typename Bookkeeper&gt;</span>
<span>__host__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1unsynchronized__pool__resource.html">thrust::mr::unsynchronized_pool_resource</a>< Upstream > & </span><span><b><a href="{{ site.baseurl }}/api/groups/group__memory__resources.html#function-tls-pool">thrust::mr::tls&#95;pool</a></b>(Upstream * upstream = NULL);</span>
</code>

## Member Classes

<h3 id="class-thrustmrdisjoint-unsynchronized-pool-resource">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1disjoint__unsynchronized__pool__resource.html">Class <code>thrust::mr::disjoint&#95;unsynchronized&#95;pool&#95;resource</code>
</a>
</h3>

**Inherits From**:
* `thrust::mr::memory_resource< Upstream::pointer >`
* `thrust::mr::validator2< Upstream, Bookkeeper >`

<h3 id="struct-thrustmrdisjoint-synchronized-pool-resource">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1disjoint__synchronized__pool__resource.html">Struct <code>thrust::mr::disjoint&#95;synchronized&#95;pool&#95;resource</code>
</a>
</h3>

**Inherits From**:
[`thrust::mr::memory_resource< Upstream::pointer >`]({{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html)

<h3 id="class-thrustmrmemory-resource">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html">Class <code>thrust::mr::memory&#95;resource</code>
</a>
</h3>

**Inherited By**:
* [`thrust::mr::new_delete_resource`]({{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1new__delete__resource.html)
* [`thrust::mr::polymorphic_adaptor_resource< Pointer >`]({{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1polymorphic__adaptor__resource.html)

<h3 id="class-thrustmrmemory-resource<-void-*->">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource_3_01void_01_5_01_4.html">Class <code>thrust::mr::memory&#95;resource&lt; void &#42; &gt;</code>
</a>
</h3>

<h3 id="class-thrustmrnew-delete-resource">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1new__delete__resource.html">Class <code>thrust::mr::new&#95;delete&#95;resource</code>
</a>
</h3>

**Inherits From**:
[`thrust::mr::memory_resource<>`]({{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html)

<h3 id="class-thrustmrunsynchronized-pool-resource">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1unsynchronized__pool__resource.html">Class <code>thrust::mr::unsynchronized&#95;pool&#95;resource</code>
</a>
</h3>

**Inherits From**:
* `thrust::mr::memory_resource< Upstream::pointer >`
* `thrust::mr::validator< Upstream >`

<h3 id="struct-thrustmrpool-options">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1pool__options.html">Struct <code>thrust::mr::pool&#95;options</code>
</a>
</h3>

<h3 id="struct-thrustmrsynchronized-pool-resource">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1synchronized__pool__resource.html">Struct <code>thrust::mr::synchronized&#95;pool&#95;resource</code>
</a>
</h3>

**Inherits From**:
[`thrust::mr::memory_resource< Upstream::pointer >`]({{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1memory__resource.html)


## Types

<h3 id="typedef-memory-resource">
Typedef <code>thrust::system::cpp::memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>typedef detail::native_resource<b>memory_resource</b>;</span></code>
The memory resource for the Standard C++ system. Uses <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1new__delete__resource.html">mr::new&#95;delete&#95;resource</a></code> and tags it with <code>cpp::pointer</code>. 

<h3 id="typedef-universal-memory-resource">
Typedef <code>thrust::system::cpp::universal&#95;memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>typedef detail::universal_native_resource<b>universal_memory_resource</b>;</span></code>
The unified memory resource for the Standard C++ system. Uses <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1new__delete__resource.html">mr::new&#95;delete&#95;resource</a></code> and tags it with <code>cpp::universal&#95;pointer</code>. 

<h3 id="typedef-universal-host-pinned-memory-resource">
Typedef <code>thrust::system::cpp::universal&#95;host&#95;pinned&#95;memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>typedef detail::native_resource<b>universal_host_pinned_memory_resource</b>;</span></code>
An alias for <code>cpp::universal&#95;memory&#95;resource</code>. 

<h3 id="typedef-memory-resource">
Typedef <code>thrust::system::omp::memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>typedef detail::native_resource<b>memory_resource</b>;</span></code>
The memory resource for the OpenMP system. Uses <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1new__delete__resource.html">mr::new&#95;delete&#95;resource</a></code> and tags it with <code>omp::pointer</code>. 

<h3 id="typedef-universal-memory-resource">
Typedef <code>thrust::system::omp::universal&#95;memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>typedef detail::universal_native_resource<b>universal_memory_resource</b>;</span></code>
The unified memory resource for the OpenMP system. Uses <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1new__delete__resource.html">mr::new&#95;delete&#95;resource</a></code> and tags it with <code>omp::universal&#95;pointer</code>. 

<h3 id="typedef-universal-host-pinned-memory-resource">
Typedef <code>thrust::system::omp::universal&#95;host&#95;pinned&#95;memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>typedef detail::native_resource<b>universal_host_pinned_memory_resource</b>;</span></code>
An alias for <code>omp::universal&#95;memory&#95;resource</code>. 

<h3 id="typedef-memory-resource">
Typedef <code>thrust::system::tbb::memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>typedef detail::native_resource<b>memory_resource</b>;</span></code>
The memory resource for the TBB system. Uses <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1new__delete__resource.html">mr::new&#95;delete&#95;resource</a></code> and tags it with <code>tbb::pointer</code>. 

<h3 id="typedef-universal-memory-resource">
Typedef <code>thrust::system::tbb::universal&#95;memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>typedef detail::universal_native_resource<b>universal_memory_resource</b>;</span></code>
The unified memory resource for the TBB system. Uses <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1new__delete__resource.html">mr::new&#95;delete&#95;resource</a></code> and tags it with <code>tbb::universal&#95;pointer</code>. 

<h3 id="typedef-universal-host-pinned-memory-resource">
Typedef <code>thrust::system::tbb::universal&#95;host&#95;pinned&#95;memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>typedef detail::native_resource<b>universal_host_pinned_memory_resource</b>;</span></code>
An alias for <code>tbb::universal&#95;memory&#95;resource</code>. 

<h3 id="using-universal-ptr">
Type Alias <code>thrust::universal&#95;ptr</code>
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
* host_ptr For the documentation of the complete interface which is shared by <code>universal&#95;ptr</code>. 
* raw_pointer_cast 


## Functions

<h3 id="function-tls-disjoint-pool">
Function <code>thrust::mr::tls&#95;disjoint&#95;pool</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Upstream,</span>
<span>&nbsp;&nbsp;typename Bookkeeper&gt;</span>
<span>__host__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1disjoint__unsynchronized__pool__resource.html">thrust::mr::disjoint_unsynchronized_pool_resource</a>< Upstream, Bookkeeper > & </span><span><b>tls_disjoint_pool</b>(Upstream * upstream = NULL,</span>
<span>&nbsp;&nbsp;Bookkeeper * bookkeeper = NULL);</span></code>
Potentially constructs, if not yet created, and then returns the address of a thread-local <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1disjoint__unsynchronized__pool__resource.html">disjoint&#95;unsynchronized&#95;pool&#95;resource</a></code>,

**Template Parameters**:
* **`Upstream`** the first template argument to the pool template 
* **`Bookkeeper`** the second template argument to the pool template 

**Function Parameters**:
* **`upstream`** the first argument to the constructor, if invoked 
* **`bookkeeper`** the second argument to the constructor, if invoked 

<h3 id="function-operator==">
Function <code>thrust::mr::operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Pointer&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const memory_resource< Pointer > & lhs,</span>
<span>&nbsp;&nbsp;const memory_resource< Pointer > & rhs);</span></code>
Compares the memory resources for equality, first by identity, then by <code>is&#95;equal</code>. 

<h3 id="function-operator!=">
Function <code>thrust::mr::operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Pointer&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const memory_resource< Pointer > & lhs,</span>
<span>&nbsp;&nbsp;const memory_resource< Pointer > & rhs);</span></code>
Compares the memory resources for inequality, first by identity, then by <code>is&#95;equal</code>. 

<h3 id="function-get-global-resource">
Function <code>thrust::mr::get&#95;global&#95;resource</code>
</h3>

<code class="doxybook">
<span>template &lt;typename MR&gt;</span>
<span>__host__ MR * </span><span><b>get_global_resource</b>();</span></code>
Returns a global instance of <code>MR</code>, created as a function local static variable.

**Template Parameters**:
**`MR`**: type of a memory resource to get an instance from. Must be <code>DefaultConstructible</code>. 

**Returns**:
a pointer to a global instance of <code>MR</code>. 

<h3 id="function-tls-pool">
Function <code>thrust::mr::tls&#95;pool</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Upstream,</span>
<span>&nbsp;&nbsp;typename Bookkeeper&gt;</span>
<span>__host__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1unsynchronized__pool__resource.html">thrust::mr::unsynchronized_pool_resource</a>< Upstream > & </span><span><b>tls_pool</b>(Upstream * upstream = NULL);</span></code>
Potentially constructs, if not yet created, and then returns the address of a thread-local <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1unsynchronized__pool__resource.html">unsynchronized&#95;pool&#95;resource</a></code>,

**Template Parameters**:
**`Upstream`**: the template argument to the pool template 

**Function Parameters**:
**`upstream`**: the argument to the constructor, if invoked 


