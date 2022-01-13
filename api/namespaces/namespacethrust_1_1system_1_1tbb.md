---
title: thrust::system::tbb
parent: Systems
grand_parent: System
nav_exclude: true
has_children: true
has_toc: false
---

# Namespace `thrust::system::tbb`

<code class="doxybook">
<span>namespace thrust::system::tbb {</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1tbb.html#using-allocator">allocator</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1tbb.html#using-universal-allocator">universal&#95;allocator</a></b> = <i>see below</i>;</span>
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
<span>&nbsp;&nbsp;typename Allocator = thrust::system::tbb::allocator&lt;T&gt;&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1tbb.html#using-vector">vector</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename Allocator = thrust::system::tbb::universal&#95;allocator&lt;T&gt;&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1tbb.html#using-universal-vector">universal&#95;vector</a></b> = <i>see below</i>;</span>
<br>
<span><a href="{{ site.baseurl }}/api/groups/group__system__backends.html#using-pointer">pointer</a>< void > </span><span><b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1tbb.html#function-malloc">malloc</a></b>(std::size_t n);</span>
<br>
<span>template &lt;typename T&gt;</span>
<span><a href="{{ site.baseurl }}/api/groups/group__system__backends.html#using-pointer">pointer</a>< T > </span><span><b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1tbb.html#function-malloc">malloc</a></b>(std::size_t n);</span>
<br>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1tbb.html#function-free">free</a></b>(<a href="{{ site.baseurl }}/api/groups/group__system__backends.html#using-pointer">pointer</a>< void > ptr);</span>
<span>} /* namespace thrust::system::tbb */</span>
</code>

## Types

<h3 id="using-allocator">
Type Alias <code>thrust::system::tbb::allocator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>allocator</b> = &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1mr&#95;1&#95;1stateless&#95;&#95;resource&#95;&#95;allocator.html"&gt;thrust::mr::stateless&#95;resource&#95;allocator&lt;/a&gt;&lt; T, &lt;a href="{{ site.baseurl }}/api/groups/group&#95;&#95;memory&#95;&#95;resources.html#typedef-memory-resource"&gt;thrust::system::tbb::memory&#95;resource&lt;/a&gt; &gt;;</span></code>
<code>tbb::allocator</code> is the default allocator used by the <code>tbb</code> system's containers such as <code>tbb::vector</code> if no user-specified allocator is provided. <code>tbb::allocator</code> allocates (deallocates) storage with <code>tbb::malloc</code> (<code>tbb::free</code>). 

<h3 id="using-universal-allocator">
Type Alias <code>thrust::system::tbb::universal&#95;allocator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>universal_allocator</b> = &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1mr&#95;1&#95;1stateless&#95;&#95;resource&#95;&#95;allocator.html"&gt;thrust::mr::stateless&#95;resource&#95;allocator&lt;/a&gt;&lt; T, &lt;a href="{{ site.baseurl }}/api/groups/group&#95;&#95;memory&#95;&#95;resources.html#typedef-universal-memory-resource"&gt;thrust::system::tbb::universal&#95;memory&#95;resource&lt;/a&gt; &gt;;</span></code>
<code>tbb::universal&#95;allocator</code> allocates memory that can be used by the <code>tbb</code> system and host systems. 

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

<h3 id="using-pointer">
Type Alias <code>thrust::system::tbb::pointer</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>pointer</b> = thrust::pointer&lt; T, thrust::system::tbb::tag, thrust::tagged&#95;reference&lt; T, thrust::system::tbb::tag &gt; &gt;;</span></code>
<code>tbb::pointer</code> stores a pointer to an object allocated in memory accessible by the <code>tbb</code> system. This type provides type safety when dispatching algorithms on ranges resident in <code>tbb</code> memory.

<code>tbb::pointer</code> has pointer semantics: it may be dereferenced and manipulated with pointer arithmetic.

<code>tbb::pointer</code> can be created with the function <code>tbb::malloc</code>, or by explicitly calling its constructor with a raw pointer.

The raw pointer encapsulated by a <code>tbb::pointer</code> may be obtained by eiter its <code>get</code> member function or the <code>raw&#95;pointer&#95;cast</code> function.

**Note**:
<code>tbb::pointer</code> is not a "smart" pointer; it is the programmer's responsibility to deallocate memory pointed to by <code>tbb::pointer</code>.

**Template Parameters**:
**`T`**: specifies the type of the pointee.

**See**:
* tbb::malloc 
* tbb::free 
* raw_pointer_cast 

<h3 id="using-universal-pointer">
Type Alias <code>thrust::system::tbb::universal&#95;pointer</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>universal_pointer</b> = thrust::pointer&lt; T, thrust::system::tbb::tag, typename std::add&#95;lvalue&#95;reference&lt; T &gt;::type &gt;;</span></code>
<code>tbb::universal&#95;pointer</code> stores a pointer to an object allocated in memory accessible by the <code>tbb</code> system and host systems.

<code>tbb::universal&#95;pointer</code> has pointer semantics: it may be dereferenced and manipulated with pointer arithmetic.

<code>tbb::universal&#95;pointer</code> can be created with <code>tbb::universal&#95;allocator</code> or by explicitly calling its constructor with a raw pointer.

The raw pointer encapsulated by a <code>tbb::universal&#95;pointer</code> may be obtained by eiter its <code>get</code> member function or the <code>raw&#95;pointer&#95;cast</code> function.

**Note**:
<code>tbb::universal&#95;pointer</code> is not a "smart" pointer; it is the programmer's responsibility to deallocate memory pointed to by <code>tbb::universal&#95;pointer</code>.

**Template Parameters**:
**`T`**: specifies the type of the pointee.

**See**:
* tbb::universal_allocator 
* raw_pointer_cast 

<h3 id="using-reference">
Type Alias <code>thrust::system::tbb::reference</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>reference</b> = thrust::tagged&#95;reference&lt; T, thrust::system::tbb::tag &gt;;</span></code>
<code>reference</code> is a wrapped reference to an object stored in memory available to the <code>tbb</code> system. <code>reference</code> is the type of the result of dereferencing a <code>tbb::pointer</code>.

**Template Parameters**:
**`T`**: Specifies the type of the referenced object. 

<h3 id="using-vector">
Type Alias <code>thrust::system::tbb::vector</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename Allocator = thrust::system::tbb::allocator&lt;T&gt;&gt;</span>
<span>using <b>vector</b> = thrust::detail::vector&#95;base&lt; T, Allocator &gt;;</span></code>
<code>tbb::vector</code> is a container that supports random access to elements, constant time removal of elements at the end, and linear time insertion and removal of elements at the beginning or in the middle. The number of elements in a <code>tbb::vector</code> may vary dynamically; memory management is automatic. The elements contained in a <code>tbb::vector</code> reside in memory accessible by the <code>tbb</code> system.

**Template Parameters**:
* **`T`** The element type of the <code>tbb::vector</code>. 
* **`Allocator`** The allocator type of the <code>tbb::vector</code>. Defaults to <code>tbb::allocator</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/container/vector">https://en.cppreference.com/w/cpp/container/vector</a>
* <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> For the documentation of the complete interface which is shared by <code>tbb::vector</code>. 
* <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device_vector</a>
* universal_vector 

<h3 id="using-universal-vector">
Type Alias <code>thrust::system::tbb::universal&#95;vector</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename Allocator = thrust::system::tbb::universal&#95;allocator&lt;T&gt;&gt;</span>
<span>using <b>universal_vector</b> = thrust::detail::vector&#95;base&lt; T, Allocator &gt;;</span></code>
<code>tbb::universal&#95;vector</code> is a container that supports random access to elements, constant time removal of elements at the end, and linear time insertion and removal of elements at the beginning or in the middle. The number of elements in a <code>tbb::universal&#95;vector</code> may vary dynamically; memory management is automatic. The elements contained in a <code>tbb::universal&#95;vector</code> reside in memory accessible by the <code>tbb</code> system and host systems.

**Template Parameters**:
* **`T`** The element type of the <code>tbb::universal&#95;vector</code>. 
* **`Allocator`** The allocator type of the <code>tbb::universal&#95;vector</code>. Defaults to <code>tbb::universal&#95;allocator</code>.

**See**:
* <a href="https://en.cppreference.com/w/cpp/container/vector">https://en.cppreference.com/w/cpp/container/vector</a>
* <a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">host_vector</a> For the documentation of the complete interface which is shared by <code>tbb::universal&#95;vector</code>
* <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device_vector</a>
* universal_vector 


## Functions

<h3 id="function-malloc">
Function <code>thrust::system::tbb::malloc</code>
</h3>

<code class="doxybook">
<span><a href="{{ site.baseurl }}/api/groups/group__system__backends.html#using-pointer">pointer</a>< void > </span><span><b>malloc</b>(std::size_t n);</span></code>
Allocates an area of memory available to Thrust's <code>tbb</code> system. 

**Note**:
The <code>tbb::pointer&lt;void&gt;</code> returned by this function must be deallocated with <code>tbb::free</code>. 

**Function Parameters**:
**`n`**: Number of bytes to allocate. 

**Returns**:
A <code>tbb::pointer&lt;void&gt;</code> pointing to the beginning of the newly allocated memory. A null <code>tbb::pointer&lt;void&gt;</code> is returned if an error occurs. 

**See**:
* tbb::free 
* std::malloc 

<h3 id="function-malloc">
Function <code>thrust::system::tbb::malloc</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span><a href="{{ site.baseurl }}/api/groups/group__system__backends.html#using-pointer">pointer</a>< T > </span><span><b>malloc</b>(std::size_t n);</span></code>
Allocates a typed area of memory available to Thrust's <code>tbb</code> system. 

**Note**:
The <code>tbb::pointer&lt;T&gt;</code> returned by this function must be deallocated with <code>tbb::free</code>. 

**Function Parameters**:
**`n`**: Number of elements to allocate. 

**Returns**:
A <code>tbb::pointer&lt;T&gt;</code> pointing to the beginning of the newly allocated memory. A null <code>tbb::pointer&lt;T&gt;</code> is returned if an error occurs. 

**See**:
* tbb::free 
* std::malloc 

<h3 id="function-free">
Function <code>thrust::system::tbb::free</code>
</h3>

<code class="doxybook">
<span>void </span><span><b>free</b>(<a href="{{ site.baseurl }}/api/groups/group__system__backends.html#using-pointer">pointer</a>< void > ptr);</span></code>
Deallocates an area of memory previously allocated by <code>tbb::malloc</code>. 

**Function Parameters**:
**`ptr`**: A <code>tbb::pointer&lt;void&gt;</code> pointing to the beginning of an area of memory previously allocated with <code>tbb::malloc</code>. 

**See**:
* tbb::malloc 
* std::free 


