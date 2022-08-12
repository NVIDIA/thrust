---
title: thrust::device_ptr
summary: device_ptr is a pointer-like object which points to an object that resides in memory associated with the device system. 
parent: Memory Management
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::device_ptr`

<code>device&#95;ptr</code> is a pointer-like object which points to an object that resides in memory associated with the device system. 

<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> has pointer semantics: it may be dereferenced safely from anywhere, including the host, and may be manipulated with pointer arithmetic.

<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> can be created with device_new, device_malloc, <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__malloc__allocator.html">device_malloc_allocator</a>, <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__allocator.html">device_allocator</a>, or device_pointer_cast, or by explicitly calling its constructor with a raw pointer.

The raw pointer contained in a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> may be obtained via <code>get</code> member function or the raw_pointer_cast free function.

<a href="{{ site.baseurl }}/api/groups/group__algorithms.html">Algorithms</a> operating on <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> types will automatically be dispatched to the device system.

**Note**:
<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> is not a smart pointer; it is the programmer's responsibility to deallocate memory pointed to by <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code>.

**Inherits From**:
`thrust::pointer< T, thrust::device_system_tag, thrust::device_reference< T >, thrust::device_ptr< T > >`

**See**:
* device_new 
* device_malloc 
* <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__malloc__allocator.html">device_malloc_allocator</a>
* <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__allocator.html">device_allocator</a>
* device_pointer_cast 
* raw_pointer_cast 

<code class="doxybook">
<span>#include <thrust/device_ptr.h></span><br>
<span>template &lt;typename T&gt;</span>
<span>class thrust::device&#95;ptr {</span>
<span>public:</span><span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Construct a null <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code>.  */</span><span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html#function-device-ptr">device&#95;ptr</a></b>();</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Construct a null <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code>.  */</span><span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html#function-device-ptr">device&#95;ptr</a></b>(std::nullptr_t);</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Construct a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> from a raw pointer which is convertible to <code>T&#42;</code>.  */</span><span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;explicit __host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html#function-device-ptr">device&#95;ptr</a></b>(U * ptr);</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Copy construct a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> from another <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> whose pointer type is convertible to <code>T&#42;</code>.  */</span><span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html#function-device-ptr">device&#95;ptr</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device_ptr</a>< U > const & other);</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Set this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> to point to the same object as another <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> whose pointer type is convertible to <code>T&#42;</code>.  */</span><span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device_ptr</a> & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html#function-operator=">operator=</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device_ptr</a>< U > const & other);</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Set this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> to null.  */</span><span>&nbsp;&nbsp;__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device_ptr</a> & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html#function-operator=">operator=</a></b>(std::nullptr_t);</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Return the raw pointer that this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> points to.  */</span><span>&nbsp;&nbsp;__host__ __device__ T * </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html#function-get">get</a></b>() const;</span>
<span>};</span>
</code>

## Member Functions

<h3 id="function-device-ptr">
Function <code>thrust::device&#95;ptr::device&#95;ptr</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>device_ptr</b>();</span></code>
Construct a null <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code>. 

**Postconditions**:
<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html#function-get">get()</a> == nullptr</code>. 

<h3 id="function-device-ptr">
Function <code>thrust::device&#95;ptr::device&#95;ptr</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>device_ptr</b>(std::nullptr_t);</span></code>
Construct a null <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code>. 

**Function Parameters**:
**`ptr`**: A null pointer.

**Postconditions**:
<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html#function-get">get()</a> == nullptr</code>. 

<h3 id="function-device-ptr">
Function <code>thrust::device&#95;ptr::device&#95;ptr</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>explicit __host__ __device__ </span><span><b>device_ptr</b>(U * ptr);</span></code>
Construct a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> from a raw pointer which is convertible to <code>T&#42;</code>. 

**Template Parameters**:
**`U`**: A type whose pointer is convertible to <code>T&#42;</code>. 

**Function Parameters**:
**`ptr`**: A raw pointer to a <code>U</code> in device memory to construct from.

**Preconditions**:
* <code>std::is&#95;convertible&#95;v&lt;U&#42;, T&#42;&gt; == true</code>.
* <code>ptr</code> points to a location in device memory.

**Postconditions**:
<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html#function-get">get()</a> == nullptr</code>. 

<h3 id="function-device-ptr">
Function <code>thrust::device&#95;ptr::device&#95;ptr</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ </span><span><b>device_ptr</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device_ptr</a>< U > const & other);</span></code>
Copy construct a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> from another <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> whose pointer type is convertible to <code>T&#42;</code>. 

**Template Parameters**:
**`U`**: A type whose pointer is convertible to <code>T&#42;</code>. 

**Function Parameters**:
**`other`**: A <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> to a <code>U</code> to construct from.

**Preconditions**:
<code>std::is&#95;convertible&#95;v&lt;U&#42;, T&#42;&gt; == true</code>.

**Postconditions**:
<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html#function-get">get()</a> == <a href="{{ site.baseurl }}/api/groups/group__tuple.html#function-get">other.get()</a></code>. 

<h3 id="function-operator=">
Function <code>thrust::device&#95;ptr::operator=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device_ptr</a> & </span><span><b>operator=</b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device_ptr</a>< U > const & other);</span></code>
Set this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> to point to the same object as another <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> whose pointer type is convertible to <code>T&#42;</code>. 

**Template Parameters**:
**`U`**: A type whose pointer is convertible to <code>T&#42;</code>. 

**Function Parameters**:
**`other`**: A <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> to a <code>U</code> to assign from.

**Preconditions**:
<code>std::is&#95;convertible&#95;v&lt;U&#42;, T&#42;&gt; == true</code>.

**Postconditions**:
<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html#function-get">get()</a> == <a href="{{ site.baseurl }}/api/groups/group__tuple.html#function-get">other.get()</a></code>.

**Returns**:
<code>&#42;this</code>. 

<h3 id="function-operator=">
Function <code>thrust::device&#95;ptr::operator=</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device_ptr</a> & </span><span><b>operator=</b>(std::nullptr_t);</span></code>
Set this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> to null. 

**Function Parameters**:
**`ptr`**: A null pointer.

**Postconditions**:
<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html#function-get">get()</a> == nullptr</code>.

**Returns**:
<code>&#42;this</code>. 

<h3 id="function-get">
Function <code>thrust::device&#95;ptr::get</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ T * </span><span><b>get</b>() const;</span></code>
Return the raw pointer that this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">device&#95;ptr</a></code> points to. 


