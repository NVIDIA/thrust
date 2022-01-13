---
title: thrust::system::omp
parent: Systems
grand_parent: System
nav_exclude: true
has_children: true
has_toc: false
---

# Namespace `thrust::system::omp`

<code class="doxybook">
<span>namespace thrust::system::omp {</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1omp.html#using-allocator">allocator</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1omp.html#using-universal-allocator">universal&#95;allocator</a></b> = <i>see below</i>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__memory__resources.html#typedef-memory-resource">memory&#95;resource</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__memory__resources.html#typedef-universal-memory-resource">universal&#95;memory&#95;resource</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__memory__resources.html#typedef-universal-host-pinned-memory-resource">universal&#95;host&#95;pinned&#95;memory&#95;resource</a></b>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__system__backends.html#using-pointer">pointer</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__system__backends.html#using-universal-pointer">universal&#95;pointer</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__system__backends.html#using-reference">reference</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename Allocator = thrust::system::omp::allocator&lt;T&gt;&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1omp.html#using-vector">vector</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename Allocator = thrust::system::omp::universal&#95;allocator&lt;T&gt;&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1omp.html#using-universal-vector">universal&#95;vector</a></b> = <i>see below</i>;</span>
<br>
<span><a href="{{ site.baseurl }}/api/groups/group__system__backends.html#using-pointer">pointer</a>< void > </span><span><b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1omp.html#function-malloc">malloc</a></b>(std::size_t n);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span><a href="{{ site.baseurl }}/api/groups/group__system__backends.html#using-pointer">pointer</a>< T > </span><span><b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1omp.html#function-malloc">malloc</a></b>(std::size_t n);</span>
<br>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1omp.html#function-free">free</a></b>(<a href="{{ site.baseurl }}/api/groups/group__system__backends.html#using-pointer">pointer</a>< void > ptr);</span>
<span>} /* namespace thrust::system::omp */</span>
</code>

## Types

<h3 id="using-allocator">
Type Alias <code>thrust::system::omp::allocator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>allocator</b> = &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1mr&#95;1&#95;1stateless&#95;&#95;resource&#95;&#95;allocator.html"&gt;thrust::mr::stateless&#95;resource&#95;allocator&lt;/a&gt;&lt; T, &lt;a href="{{ site.baseurl }}/api/groups/group&#95;&#95;memory&#95;&#95;resources.html#typedef-memory-resource"&gt;thrust::system::omp::memory&#95;resource&lt;/a&gt; &gt;;</span></code>
<code>omp::allocator</code> is the default allocator used by the <code>omp</code> system's containers such as <code>omp::vector</code> if no user-specified allocator is provided. <code>omp::allocator</code> allocates (deallocates) storage with <code>omp::malloc</code> (<code>omp::free</code>). 

<h3 id="using-universal-allocator">
Type Alias <code>thrust::system::omp::universal&#95;allocator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>universal_allocator</b> = &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1mr&#95;1&#95;1stateless&#95;&#95;resource&#95;&#95;allocator.html"&gt;thrust::mr::stateless&#95;resource&#95;allocator&lt;/a&gt;&lt; T, &lt;a href="{{ site.baseurl }}/api/groups/group&#95;&#95;memory&#95;&#95;resources.html#typedef-universal-memory-resource"&gt;thrust::system::omp::universal&#95;memory&#95;resource&lt;/a&gt; &gt;;</span></code>
<code>omp::universal&#95;allocator</code> allocates memory that can be used by the <code>omp</code> system and host systems. 

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

<h3 id="using-pointer">
Type Alias <code>thrust::system::omp::pointer</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>pointer</b> = thrust::pointer&lt; T, thrust::system::omp::tag, thrust::tagged&#95;reference&lt; T, thrust::system::omp::tag &gt; &gt;;</span></code>
<code>omp::pointer</code> stores a pointer to an object allocated in memory accessible by the <code>omp</code> system. This type provides type safety when dispatching algorithms on ranges resident in <code>omp</code> memory.

<code>omp::pointer</code> has pointer semantics: it may be dereferenced and manipulated with pointer arithmetic.

<code>omp::pointer</code> can be created with the function <code>omp::malloc</code>, or by explicitly calling its constructor with a raw pointer.

The raw pointer encapsulated by a <code>omp::pointer</code> may be obtained by eiter its <code>get</code> member function or the <code>raw&#95;pointer&#95;cast</code> function.

**Note**:
<code>omp::pointer</code> is not a "smart" pointer; it is the programmer's responsibility to deallocate memory pointed to by <code>omp::pointer</code>.

**Template Parameters**:
**`T`**: specifies the type of the pointee.

**See**:
* omp::malloc 
* omp::free 
* raw_pointer_cast 

<h3 id="using-universal-pointer">
Type Alias <code>thrust::system::omp::universal&#95;pointer</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>universal_pointer</b> = thrust::pointer&lt; T, thrust::system::omp::tag, typename std::add&#95;lvalue&#95;reference&lt; T &gt;::type &gt;;</span></code>
<code>omp::universal&#95;pointer</code> stores a pointer to an object allocated in memory accessible by the <code>omp</code> system and host systems.

<code>omp::universal&#95;pointer</code> has pointer semantics: it may be dereferenced and manipulated with pointer arithmetic.

<code>omp::universal&#95;pointer</code> can be created with <code>omp::universal&#95;allocator</code> or by explicitly calling its constructor with a raw pointer.

The raw pointer encapsulated by a <code>omp::universal&#95;pointer</code> may be obtained by eiter its <code>get</code> member function or the <code>raw&#95;pointer&#95;cast</code> function.

**Note**:
<code>omp::universal&#95;pointer</code> is not a "smart" pointer; it is the programmer's responsibility to deallocate memory pointed to by <code>omp::universal&#95;pointer</code>.

**Template Parameters**:
**`T`**: specifies the type of the pointee.

**See**:
* omp::universal_allocator 
* raw_pointer_cast 

<h3 id="using-reference">
Type Alias <code>thrust::system::omp::reference</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>reference</b> = thrust::tagged&#95;reference&lt; T, thrust::system::omp::tag &gt;;</span></code>
<code>reference</code> is a wrapped reference to an object stored in memory available to the <code>omp</code> system. <code>reference</code> is the type of the result of dereferencing a <code>omp::pointer</code>.

**Template Parameters**:
**`T`**: Specifies the type of the referenced object. 

<h3 id="using-vector">
Type Alias <code>thrust::system::omp::vector</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename Allocator = thrust::system::omp::allocator&lt;T&gt;&gt;</span>
<span>using <b>vector</b> = thrust::detail::vector&#95;base&lt; T, Allocator &gt;;</span></code>
<code>omp::vector</code> is a container that supports random access to elements, constant time removal of elements at the end, and linear time insertion and removal of elements at the beginning or in the middle. The number of elements in a <code>omp::vector</code> may vary dynamically; memory management is automatic. The elements contained in an <code>omp::vector</code> reside in memory accessible by the <code>omp</code> system.

**Template Parameters**:
* **`T`** The element type of the <code>omp::vector</code>. 
* **`Allocator`** The allocator type of the <code>omp::vector</code>. Defaults to <code>omp::allocator</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/container/vector">https://en.cppreference.com/w/cpp/container/vector</a>
* <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> For the documentation of the complete interface which is shared by <code>omp::vector</code>. 
* <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device_vector</a>
* universal_vector 

<h3 id="using-universal-vector">
Type Alias <code>thrust::system::omp::universal&#95;vector</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename Allocator = thrust::system::omp::universal&#95;allocator&lt;T&gt;&gt;</span>
<span>using <b>universal_vector</b> = thrust::detail::vector&#95;base&lt; T, Allocator &gt;;</span></code>
<code>omp::universal&#95;vector</code> is a container that supports random access to elements, constant time removal of elements at the end, and linear time insertion and removal of elements at the beginning or in the middle. The number of elements in a <code>omp::universal&#95;vector</code> may vary dynamically; memory management is automatic. The elements contained in a <code>omp::universal&#95;vector</code> reside in memory accessible by the <code>omp</code> system and host systems.

**Template Parameters**:
* **`T`** The element type of the <code>omp::universal&#95;vector</code>. 
* **`Allocator`** The allocator type of the <code>omp::universal&#95;vector</code>. Defaults to <code>omp::universal&#95;allocator</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/container/vector">https://en.cppreference.com/w/cpp/container/vector</a>
* <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> For the documentation of the complete interface which is shared by <code>omp::universal&#95;vector</code>
* <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device_vector</a>
* universal_vector 


## Functions

<h3 id="function-malloc">
Function <code>thrust::system::omp::malloc</code>
</h3>

<code class="doxybook">
<span><a href="{{ site.baseurl }}/api/groups/group__system__backends.html#using-pointer">pointer</a>< void > </span><span><b>malloc</b>(std::size_t n);</span></code>
Allocates an area of memory available to Thrust's <code>omp</code> system. 

**Note**:
The <code>omp::pointer&lt;void&gt;</code> returned by this function must be deallocated with <code>omp::free</code>. 

**Function Parameters**:
**`n`**: Number of bytes to allocate. 

**Returns**:
A <code>omp::pointer&lt;void&gt;</code> pointing to the beginning of the newly allocated memory. A null <code>omp::pointer&lt;void&gt;</code> is returned if an error occurs. 

**See**:
* omp::free 
* std::malloc 

<h3 id="function-malloc">
Function <code>thrust::system::omp::malloc</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span><a href="{{ site.baseurl }}/api/groups/group__system__backends.html#using-pointer">pointer</a>< T > </span><span><b>malloc</b>(std::size_t n);</span></code>
Allocates a typed area of memory available to Thrust's <code>omp</code> system. 

**Note**:
The <code>omp::pointer&lt;T&gt;</code> returned by this function must be deallocated with <code>omp::free</code>. 

**Function Parameters**:
**`n`**: Number of elements to allocate. 

**Returns**:
A <code>omp::pointer&lt;T&gt;</code> pointing to the beginning of the newly allocated memory. A null <code>omp::pointer&lt;T&gt;</code> is returned if an error occurs. 

**See**:
* omp::free 
* std::malloc 

<h3 id="function-free">
Function <code>thrust::system::omp::free</code>
</h3>

<code class="doxybook">
<span>void </span><span><b>free</b>(<a href="{{ site.baseurl }}/api/groups/group__system__backends.html#using-pointer">pointer</a>< void > ptr);</span></code>
Deallocates an area of memory previously allocated by <code>omp::malloc</code>. 

**Function Parameters**:
**`ptr`**: A <code>omp::pointer&lt;void&gt;</code> pointing to the beginning of an area of memory previously allocated with <code>omp::malloc</code>. 

**See**:
* omp::malloc 
* std::free 


