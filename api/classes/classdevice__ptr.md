---
title: device_ptr
summary: device_ptr is a pointer-like object which points to an object that resides in memory associated with the device system. 
parent: Memory Management
nav_exclude: true
has_children: true
has_toc: false
---

# Class `device_ptr`

<code>device&#95;ptr</code> is a pointer-like object which points to an object that resides in memory associated with the device system. 

<code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code> has pointer semantics: it may be dereferenced safely from anywhere, including the <a href="/thrust/api/groups/group__execution__policies.html#variable-host">host</a>, and may be manipulated with pointer arithmetic.

<code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code> can be created with <a href="/thrust/api/groups/group__memory__management.html#function-device_new">device_new</a>, <a href="/thrust/api/groups/group__memory__management.html#function-device_malloc">device_malloc</a>, <a href="/thrust/api/classes/classdevice__malloc__allocator.html">device_malloc_allocator</a>, <a href="/thrust/api/classes/classdevice__allocator.html">device_allocator</a>, or <a href="/thrust/api/groups/group__memory__management.html#function-device_pointer_cast">device_pointer_cast</a>, or by explicitly calling its constructor with a raw pointer.

The raw pointer contained in a <code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code> may be obtained via <code>get</code> member function or the <a href="/thrust/api/groups/group__memory__management.html#function-raw_pointer_cast">raw_pointer_cast</a> free function.

<a href="/thrust/api/groups/group__algorithms.html">Algorithms</a> operating on <code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code> types will automatically be dispatched to the <a href="/thrust/api/groups/group__execution__policies.html#variable-device">device</a> system.

**Note**:
<code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code> is not a smart pointer; it is the programmer's responsibility to deallocate memory pointed to by <code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code>.

**See**:
* <a href="/thrust/api/groups/group__memory__management.html#function-device_new">device_new</a>
* <a href="/thrust/api/groups/group__memory__management.html#function-device_malloc">device_malloc</a>
* <a href="/thrust/api/classes/classdevice__malloc__allocator.html">device_malloc_allocator</a>
* <a href="/thrust/api/classes/classdevice__allocator.html">device_allocator</a>
* <a href="/thrust/api/groups/group__memory__management.html#function-device_pointer_cast">device_pointer_cast</a>
* <a href="/thrust/api/groups/group__memory__management.html#function-raw_pointer_cast">raw_pointer_cast</a>

<code class="doxybook">
<span>#include <thrust/device_ptr.h></span><br>
<span>template &lt;typename T&gt;</span>
<span>class device&#95;ptr {</span>
<span>public:</span><span class="doxybook-comment">&nbsp;&nbsp;/* Construct a null <code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code>.  */</span><span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classdevice__ptr.html#function-device_ptr">device&#95;ptr</a></b>();</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Construct a null <code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code>.  */</span><span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classdevice__ptr.html#function-device_ptr">device&#95;ptr</a></b>(std::nullptr_t ptr);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Construct a <code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code> from a raw pointer which is convertible to <code>T&#42;</code>.  */</span><span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;explicit __host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classdevice__ptr.html#function-device_ptr">device&#95;ptr</a></b>(U * ptr);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Copy construct a <code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code> from another <code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code> whose pointer type is convertible to <code>T&#42;</code>.  */</span><span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classdevice__ptr.html#function-device_ptr">device&#95;ptr</a></b>(<a href="/thrust/api/classes/classdevice__ptr.html">device_ptr</a>< U > const & other);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Set this <code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code> to point to the same object as another <code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code> whose pointer type is convertible to <code>T&#42;</code>.  */</span><span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ <a href="/thrust/api/classes/classdevice__ptr.html">device_ptr</a> & </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classdevice__ptr.html#function-operator=">operator=</a></b>(<a href="/thrust/api/classes/classdevice__ptr.html">device_ptr</a>< U > const & other);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Set this <code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code> to null.  */</span><span>&nbsp;&nbsp;__host__ __device__ <a href="/thrust/api/classes/classdevice__ptr.html">device_ptr</a> & </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classdevice__ptr.html#function-operator=">operator=</a></b>(std::nullptr_t ptr);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Return the raw pointer that this <code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code> points to.  */</span><span>&nbsp;&nbsp;__host__ __device__ T * </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classdevice__ptr.html#function-get">get</a></b>() const;</span>
<span>};</span>
</code>

## Member Functions

<h3 id="function-device_ptr">
Function <code>device&#95;ptr::&gt;::device&#95;ptr</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>device_ptr</b>();</span></code>
Construct a null <code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code>. 

**Postconditions**:
<code><a href="/thrust/api/classes/classdevice__ptr.html#function-get">get()</a> == nullptr</code>. 

<h3 id="function-device_ptr">
Function <code>device&#95;ptr::&gt;::device&#95;ptr</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>device_ptr</b>(std::nullptr_t ptr);</span></code>
Construct a null <code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code>. 

**Function Parameters**:
**`ptr`**: A null pointer.

**Postconditions**:
<code><a href="/thrust/api/classes/classdevice__ptr.html#function-get">get()</a> == nullptr</code>. 

<h3 id="function-device_ptr">
Function <code>device&#95;ptr::&gt;::device&#95;ptr</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>explicit __host__ __device__ </span><span><b>device_ptr</b>(U * ptr);</span></code>
Construct a <code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code> from a raw pointer which is convertible to <code>T&#42;</code>. 

**Template Parameters**:
**`U`**: A type whose pointer is convertible to <code>T&#42;</code>. 

**Function Parameters**:
**`ptr`**: A raw pointer to a <code>U</code> in device memory to construct from.

**Preconditions**:
* <code>std::is&#95;convertible&#95;v&lt;U&#42;, T&#42;&gt; == true</code>.
* <code>ptr</code> points to a location in device memory.

**Postconditions**:
<code><a href="/thrust/api/classes/classdevice__ptr.html#function-get">get()</a> == nullptr</code>. 

<h3 id="function-device_ptr">
Function <code>device&#95;ptr::&gt;::device&#95;ptr</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ </span><span><b>device_ptr</b>(<a href="/thrust/api/classes/classdevice__ptr.html">device_ptr</a>< U > const & other);</span></code>
Copy construct a <code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code> from another <code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code> whose pointer type is convertible to <code>T&#42;</code>. 

**Template Parameters**:
**`U`**: A type whose pointer is convertible to <code>T&#42;</code>. 

**Function Parameters**:
**`other`**: A <code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code> to a <code>U</code> to construct from.

**Preconditions**:
<code>std::is&#95;convertible&#95;v&lt;U&#42;, T&#42;&gt; == true</code>.

**Postconditions**:
<code><a href="/thrust/api/classes/classdevice__ptr.html#function-get">get()</a> == other.get()</code>. 

<h3 id="function-operator=">
Function <code>device&#95;ptr::&gt;::operator=</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ <a href="/thrust/api/classes/classdevice__ptr.html">device_ptr</a> & </span><span><b>operator=</b>(<a href="/thrust/api/classes/classdevice__ptr.html">device_ptr</a>< U > const & other);</span></code>
Set this <code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code> to point to the same object as another <code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code> whose pointer type is convertible to <code>T&#42;</code>. 

**Template Parameters**:
**`U`**: A type whose pointer is convertible to <code>T&#42;</code>. 

**Function Parameters**:
**`other`**: A <code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code> to a <code>U</code> to assign from.

**Preconditions**:
<code>std::is&#95;convertible&#95;v&lt;U&#42;, T&#42;&gt; == true</code>.

**Postconditions**:
<code><a href="/thrust/api/classes/classdevice__ptr.html#function-get">get()</a> == other.get()</code>.

**Returns**:
<code>&#42;this</code>. 

<h3 id="function-operator=">
Function <code>device&#95;ptr::&gt;::operator=</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ <a href="/thrust/api/classes/classdevice__ptr.html">device_ptr</a> & </span><span><b>operator=</b>(std::nullptr_t ptr);</span></code>
Set this <code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code> to null. 

**Function Parameters**:
**`ptr`**: A null pointer.

**Postconditions**:
<code><a href="/thrust/api/classes/classdevice__ptr.html#function-get">get()</a> == nullptr</code>.

**Returns**:
<code>&#42;this</code>. 

<h3 id="function-get">
Function <code>device&#95;ptr::&gt;::get</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ T * </span><span><b>get</b>() const;</span></code>
Return the raw pointer that this <code><a href="/thrust/api/classes/classdevice__ptr.html">device&#95;ptr</a></code> points to. 


