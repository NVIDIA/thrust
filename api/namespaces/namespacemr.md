---
title: mr
summary: thrust::mr is the namespace containing system agnostic types and functions for memory_resource related functionalities. 
nav_exclude: true
has_children: true
has_toc: false
---

# Namespace `mr`

<code class="doxybook">
<span>namespace mr {</span>
<br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;class MR&gt;</span>
<span>class <b><a href="/api/classes/classmr_1_1allocator.html">allocator</a></b>;</span>
<br>
<span>template &lt;typename Upstream,</span>
<span>&nbsp;&nbsp;typename Bookkeeper&gt;</span>
<span>struct <b><a href="/api/classes/structmr_1_1disjoint__synchronized__pool__resource.html">disjoint&#95;synchronized&#95;pool&#95;resource</a></b>;</span>
<br>
<span>template &lt;typename Upstream,</span>
<span>&nbsp;&nbsp;typename Bookkeeper&gt;</span>
<span>class <b><a href="/api/classes/classmr_1_1disjoint__unsynchronized__pool__resource.html">disjoint&#95;unsynchronized&#95;pool&#95;resource</a></b>;</span>
<br>
<span>template &lt;typename Upstream,</span>
<span>&nbsp;&nbsp;typename Pointer&gt;</span>
<span>class <b><a href="/api/classes/classmr_1_1fancy__pointer__resource.html">fancy&#95;pointer&#95;resource</a></b>;</span>
<br>
<span>template &lt;typename Pointer = void &#42;&gt;</span>
<span>class <b><a href="/api/classes/classmr_1_1memory__resource.html">memory&#95;resource</a></b>;</span>
<br>
<span>class <b><a href="/api/classes/classmr_1_1memory__resource_3_01void_01_5_01_4.html">memory&#95;resource&lt; void &#42; &gt;</a></b>;</span>
<br>
<span>class <b><a href="/api/classes/classmr_1_1new__delete__resource.html">new&#95;delete&#95;resource</a></b>;</span>
<br>
<span>template &lt;typename Pointer = void &#42;&gt;</span>
<span>class <b><a href="/api/classes/classmr_1_1polymorphic__adaptor__resource.html">polymorphic&#95;adaptor&#95;resource</a></b>;</span>
<br>
<span>struct <b><a href="/api/classes/structmr_1_1pool__options.html">pool&#95;options</a></b>;</span>
<br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename Upstream&gt;</span>
<span>class <b><a href="/api/classes/classmr_1_1stateless__resource__allocator.html">stateless&#95;resource&#95;allocator</a></b>;</span>
<br>
<span>template &lt;typename Upstream&gt;</span>
<span>struct <b><a href="/api/classes/structmr_1_1synchronized__pool__resource.html">synchronized&#95;pool&#95;resource</a></b>;</span>
<br>
<span>template &lt;typename Upstream&gt;</span>
<span>class <b><a href="/api/classes/classmr_1_1unsynchronized__pool__resource.html">unsynchronized&#95;pool&#95;resource</a></b>;</span>
<br>
<span>template &lt;typename MR&gt;</span>
<span>struct <b><a href="/api/classes/structmr_1_1validator.html">validator</a></b>;</span>
<br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename U&gt;</span>
<span>struct <b><a href="/api/classes/structmr_1_1validator2.html">validator2</a></b>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>struct <b><a href="/api/classes/structmr_1_1validator2_3_01t_00_01t_01_4.html">validator2&lt; T, T &gt;</a></b>;</span>
<br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename Pointer&gt;</span>
<span>using <b><a href="/api/groups/group__allocators.html#using-polymorphic_allocator">polymorphic&#95;allocator</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename MR&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="/api/groups/group__allocators.html#function-operator==">operator==</a></b>(const <a href="/api/classes/classmr_1_1allocator.html">allocator</a>< T, MR > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classmr_1_1allocator.html">allocator</a>< T, MR > & rhs);</span>
<br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename MR&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="/api/groups/group__allocators.html#function-operator!=">operator!=</a></b>(const <a href="/api/classes/classmr_1_1allocator.html">allocator</a>< T, MR > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classmr_1_1allocator.html">allocator</a>< T, MR > & rhs);</span>
<br>
<span>template &lt;typename Upstream,</span>
<span>&nbsp;&nbsp;typename Bookkeeper&gt;</span>
<span>__host__ thrust::mr::disjoint_unsynchronized_pool_resource< Upstream, Bookkeeper > & </span><span><b><a href="/api/groups/group__memory__resources.html#function-tls_disjoint_pool">tls&#95;disjoint&#95;pool</a></b>(Upstream * upstream = NULL,</span>
<span>&nbsp;&nbsp;Bookkeeper * bookkeeper = NULL);</span>
<br>
<span>template &lt;typename Pointer&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="/api/groups/group__memory__resources.html#function-operator==">operator==</a></b>(const <a href="/api/classes/classmr_1_1memory__resource.html">memory_resource</a>< Pointer > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classmr_1_1memory__resource.html">memory_resource</a>< Pointer > & rhs);</span>
<br>
<span>template &lt;typename Pointer&gt;</span>
<span>__host__ __device__ bool </span><span><b><a href="/api/groups/group__memory__resources.html#function-operator!=">operator!=</a></b>(const <a href="/api/classes/classmr_1_1memory__resource.html">memory_resource</a>< Pointer > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classmr_1_1memory__resource.html">memory_resource</a>< Pointer > & rhs);</span>
<br>
<span>template &lt;typename MR&gt;</span>
<span>__host__ MR * </span><span><b><a href="/api/groups/group__memory__resources.html#function-get_global_resource">get&#95;global&#95;resource</a></b>();</span>
<br>
<span>template &lt;typename Upstream,</span>
<span>&nbsp;&nbsp;typename Bookkeeper&gt;</span>
<span>__host__ thrust::mr::unsynchronized_pool_resource< Upstream > & </span><span><b><a href="/api/groups/group__memory__resources.html#function-tls_pool">tls&#95;pool</a></b>(Upstream * upstream = NULL);</span>
<span>} /* namespace mr */</span>
</code>

## Member Classes

<h3 id="class-mr::allocator">
<a href="/api/classes/classmr_1_1allocator.html">Class <code>mr::allocator</code>
</a>
</h3>

**Inherits From**:
`mr::validator< MR >`

<h3 id="struct-mr::disjoint_synchronized_pool_resource">
<a href="/api/classes/structmr_1_1disjoint__synchronized__pool__resource.html">Struct <code>mr::disjoint&#95;synchronized&#95;pool&#95;resource</code>
</a>
</h3>

**Inherits From**:
[`mr::memory_resource< Upstream::pointer >`](/api/classes/classmr_1_1memory__resource.html)

<h3 id="class-mr::disjoint_unsynchronized_pool_resource">
<a href="/api/classes/classmr_1_1disjoint__unsynchronized__pool__resource.html">Class <code>mr::disjoint&#95;unsynchronized&#95;pool&#95;resource</code>
</a>
</h3>

**Inherits From**:
* `mr::memory_resource< Upstream::pointer >`
* `mr::validator2< Upstream, Bookkeeper >`

<h3 id="class-mr::fancy_pointer_resource">
<a href="/api/classes/classmr_1_1fancy__pointer__resource.html">Class <code>mr::fancy&#95;pointer&#95;resource</code>
</a>
</h3>

**Inherits From**:
* `mr::memory_resource< Pointer >`
* `mr::validator< Upstream >`

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

<h3 id="class-mr::polymorphic_adaptor_resource">
<a href="/api/classes/classmr_1_1polymorphic__adaptor__resource.html">Class <code>mr::polymorphic&#95;adaptor&#95;resource</code>
</a>
</h3>

**Inherits From**:
[`mr::memory_resource< void * >`](/api/classes/classmr_1_1memory__resource.html)

<h3 id="struct-mr::pool_options">
<a href="/api/classes/structmr_1_1pool__options.html">Struct <code>mr::pool&#95;options</code>
</a>
</h3>

<h3 id="class-mr::stateless_resource_allocator">
<a href="/api/classes/classmr_1_1stateless__resource__allocator.html">Class <code>mr::stateless&#95;resource&#95;allocator</code>
</a>
</h3>

**Inherits From**:
`thrust::mr::allocator< T, Upstream >`

<h3 id="struct-mr::synchronized_pool_resource">
<a href="/api/classes/structmr_1_1synchronized__pool__resource.html">Struct <code>mr::synchronized&#95;pool&#95;resource</code>
</a>
</h3>

**Inherits From**:
[`mr::memory_resource< Upstream::pointer >`](/api/classes/classmr_1_1memory__resource.html)

<h3 id="class-mr::unsynchronized_pool_resource">
<a href="/api/classes/classmr_1_1unsynchronized__pool__resource.html">Class <code>mr::unsynchronized&#95;pool&#95;resource</code>
</a>
</h3>

**Inherits From**:
* `mr::memory_resource< Upstream::pointer >`
* `mr::validator< Upstream >`

<h3 id="struct-mr::validator">
<a href="/api/classes/structmr_1_1validator.html">Struct <code>mr::validator</code>
</a>
</h3>

**Inherited By**:
* [`mr::allocator< T, MR >`](/api/classes/classmr_1_1allocator.html)
* [`mr::fancy_pointer_resource< Upstream, Pointer >`](/api/classes/classmr_1_1fancy__pointer__resource.html)
* [`mr::unsynchronized_pool_resource< Upstream >`](/api/classes/classmr_1_1unsynchronized__pool__resource.html)
* [`mr::validator2< T, U >`](/api/classes/structmr_1_1validator2.html)
* [`mr::validator2< T, U >`](/api/classes/structmr_1_1validator2.html)
* [`mr::validator2< T, T >`](/api/classes/structmr_1_1validator2_3_01t_00_01t_01_4.html)
* [`mr::validator2< Upstream, Bookkeeper >`](/api/classes/structmr_1_1validator2.html)
* [`mr::validator2< Upstream, Bookkeeper >`](/api/classes/structmr_1_1validator2.html)

<h3 id="struct-mr::validator2">
<a href="/api/classes/structmr_1_1validator2.html">Struct <code>mr::validator2</code>
</a>
</h3>

**Inherits From**:
* `mr::validator< T >`
* `mr::validator< U >`

**Inherited By**:
[`mr::disjoint_unsynchronized_pool_resource< Upstream, Bookkeeper >`](/api/classes/classmr_1_1disjoint__unsynchronized__pool__resource.html)

<h3 id="struct-mr::validator2< T, T >">
<a href="/api/classes/structmr_1_1validator2_3_01t_00_01t_01_4.html">Struct <code>mr::validator2&lt; T, T &gt;</code>
</a>
</h3>

**Inherits From**:
`mr::validator< T >`


## Types

<h3 id="using-polymorphic_allocator">
Type Alias <code>polymorphic&#95;allocator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename Pointer&gt;</span>
<span>using <b>polymorphic_allocator</b> = &lt;a href="/api/classes/classmr&#95;1&#95;1allocator.html"&gt;allocator&lt;/a&gt;&lt; T, polymorphic&#95;adaptor&#95;resource&lt; Pointer &gt; &gt;;</span></code>

## Functions

<h3 id="function-operator==">
Function <code>mr::operator==</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename MR&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator==</b>(const <a href="/api/classes/classmr_1_1allocator.html">allocator</a>< T, MR > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classmr_1_1allocator.html">allocator</a>< T, MR > & rhs);</span></code>
Compares the allocators for equality by comparing the underlying memory resources. 

<h3 id="function-operator!=">
Function <code>mr::operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename MR&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const <a href="/api/classes/classmr_1_1allocator.html">allocator</a>< T, MR > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classmr_1_1allocator.html">allocator</a>< T, MR > & rhs);</span></code>
Compares the allocators for inequality by comparing the underlying memory resources. 

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
<span>__host__ __device__ bool </span><span><b>operator==</b>(const <a href="/api/classes/classmr_1_1memory__resource.html">memory_resource</a>< Pointer > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classmr_1_1memory__resource.html">memory_resource</a>< Pointer > & rhs);</span></code>
Compares the memory resources for equality, first by identity, then by <code>is&#95;equal</code>. 

<h3 id="function-operator!=">
Function <code>mr::operator!=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Pointer&gt;</span>
<span>__host__ __device__ bool </span><span><b>operator!=</b>(const <a href="/api/classes/classmr_1_1memory__resource.html">memory_resource</a>< Pointer > & lhs,</span>
<span>&nbsp;&nbsp;const <a href="/api/classes/classmr_1_1memory__resource.html">memory_resource</a>< Pointer > & rhs);</span></code>
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


