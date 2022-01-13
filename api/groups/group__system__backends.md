---
title: Systems
parent: System
grand_parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Systems

<code class="doxybook">
<span class="doxybook-comment">/* <code>thrust::cpp</code> is a top-level alias for thrust::system::cpp.  */</span><span>namespace <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1cpp.html">thrust::cpp</a></b> { <i>…</i> }</span>
<br>
<span class="doxybook-comment">/* <code>thrust::system</code> is the namespace which contains specific Thrust backend systems. It also contains functionality for reporting error conditions originating from the operating system or other low-level application program interfaces such as the CUDA runtime. They are provided in a separate namespace for import convenience but are also aliased in the top-level <code>thrust</code> namespace for easy access.  */</span><span>namespace <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system.html">thrust::system</a></b> { <i>…</i> }</span>
<br>
<span>namespace <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1omp.html">thrust::system::omp</a></b> { <i>…</i> }</span>
<br>
<span>namespace <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1cpp.html">thrust::system::cpp</a></b> { <i>…</i> }</span>
<br>
<span class="doxybook-comment">/* <code>thrust::system::cuda</code> is the namespace containing functionality for allocating, manipulating, and deallocating memory available to Thrust's CUDA backend system. The identifiers are provided in a separate namespace underneath <code><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system.html">thrust::system</a></code> for import convenience but are also aliased in the top-level <code><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1cuda.html">thrust::cuda</a></code> namespace for easy access.  */</span><span>namespace <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1cuda.html">thrust::system::cuda</a></b> { <i>…</i> }</span>
<br>
<span>namespace <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1tbb.html">thrust::system::tbb</a></b> { <i>…</i> }</span>
<br>
<span class="doxybook-comment">/* <code>thrust::omp</code> is a top-level alias for thrust::system::omp.  */</span><span>namespace <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1omp.html">thrust::omp</a></b> { <i>…</i> }</span>
<br>
<span class="doxybook-comment">/* <code>thrust::tbb</code> is a top-level alias for thrust::system::tbb.  */</span><span>namespace <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1tbb.html">thrust::tbb</a></b> { <i>…</i> }</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__system__backends.html#using-pointer">thrust::system::cpp::pointer</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__system__backends.html#using-universal-pointer">thrust::system::cpp::universal&#95;pointer</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__system__backends.html#using-reference">thrust::system::cpp::reference</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__system__backends.html#using-pointer">thrust::system::omp::pointer</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__system__backends.html#using-universal-pointer">thrust::system::omp::universal&#95;pointer</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__system__backends.html#using-reference">thrust::system::omp::reference</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__system__backends.html#using-pointer">thrust::system::tbb::pointer</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__system__backends.html#using-universal-pointer">thrust::system::tbb::universal&#95;pointer</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__system__backends.html#using-reference">thrust::system::tbb::reference</a></b> = <i>see below</i>;</span>
</code>

## Types

<h3 id="using-pointer">
Type Alias <code>thrust::system::cpp::pointer</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>pointer</b> = thrust::pointer&lt; T, thrust::system::cpp::tag, thrust::tagged&#95;reference&lt; T, thrust::system::cpp::tag &gt; &gt;;</span></code>
<code>cpp::pointer</code> stores a pointer to an object allocated in memory accessible by the <code>cpp</code> system. This type provides type safety when dispatching algorithms on ranges resident in <code>cpp</code> memory.

<code>cpp::pointer</code> has pointer semantics: it may be dereferenced and manipulated with pointer arithmetic.

<code>cpp::pointer</code> can be created with the function <code>cpp::malloc</code>, or by explicitly calling its constructor with a raw pointer.

The raw pointer encapsulated by a <code>cpp::pointer</code> may be obtained by eiter its <code>get</code> member function or the <code>raw&#95;pointer&#95;cast</code> function.

**Note**:
<code>cpp::pointer</code> is not a "smart" pointer; it is the programmer's responsibility to deallocate memory pointed to by <code>cpp::pointer</code>.

**Template Parameters**:
**`T`**: specifies the type of the pointee.

**See**:
* cpp::malloc 
* cpp::free 
* raw_pointer_cast 

<h3 id="using-universal-pointer">
Type Alias <code>thrust::system::cpp::universal&#95;pointer</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>universal_pointer</b> = thrust::pointer&lt; T, thrust::system::cpp::tag, typename std::add&#95;lvalue&#95;reference&lt; T &gt;::type &gt;;</span></code>
<code>cpp::universal&#95;pointer</code> stores a pointer to an object allocated in memory accessible by the <code>cpp</code> system and host systems.

<code>cpp::universal&#95;pointer</code> has pointer semantics: it may be dereferenced and manipulated with pointer arithmetic.

<code>cpp::universal&#95;pointer</code> can be created with <code>cpp::universal&#95;allocator</code> or by explicitly calling its constructor with a raw pointer.

The raw pointer encapsulated by a <code>cpp::universal&#95;pointer</code> may be obtained by eiter its <code>get</code> member function or the <code>raw&#95;pointer&#95;cast</code> function.

**Note**:
<code>cpp::universal&#95;pointer</code> is not a "smart" pointer; it is the programmer's responsibility to deallocate memory pointed to by <code>cpp::universal&#95;pointer</code>.

**Template Parameters**:
**`T`**: specifies the type of the pointee.

**See**:
* cpp::universal_allocator 
* raw_pointer_cast 

<h3 id="using-reference">
Type Alias <code>thrust::system::cpp::reference</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>reference</b> = thrust::reference&lt; T, thrust::system::cpp::tag &gt;;</span></code>
<code>reference</code> is a wrapped reference to an object stored in memory available to the <code>cpp</code> system. <code>reference</code> is the type of the result of dereferencing a <code>cpp::pointer</code>.

**Template Parameters**:
**`T`**: Specifies the type of the referenced object. 

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


