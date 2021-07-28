---
title: device_new_allocator
parent: Allocators
grand_parent: Memory Management
nav_exclude: true
has_children: true
has_toc: false
---

# Class `device_new_allocator`

<code><a href="/api/classes/classdevice__new__allocator.html">device&#95;new&#95;allocator</a></code> is a device memory allocator that employs the <code>device&#95;new</code> function for allocation.

**See**:
* <a href="/api/groups/group__memory__management.html#function-device_new">device_new</a>
* <a href="/api/classes/classdevice__ptr.html">device_ptr</a>
* <a href="https://en.cppreference.com/w/cpp/memory/allocator">https://en.cppreference.com/w/cpp/memory/allocator</a>

<code class="doxybook">
<span>#include <thrust/device_new_allocator.h></span><br>
<span>template &lt;typename T&gt;</span>
<span>class device&#95;new&#95;allocator {</span>
<span>public:</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="/api/classes/classdevice__new__allocator.html#typedef-value_type">value&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="/api/classes/classdevice__new__allocator.html#typedef-pointer">pointer</a></b>;</span>
<br>
<span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="/api/classes/classdevice__new__allocator.html#typedef-const_pointer">const&#95;pointer</a></b>;</span>
<br>
<span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="/api/classes/classdevice__new__allocator.html#typedef-reference">reference</a></b>;</span>
<br>
<span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="/api/classes/classdevice__new__allocator.html#typedef-const_reference">const&#95;reference</a></b>;</span>
<br>
<span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="/api/classes/classdevice__new__allocator.html#typedef-size_type">size&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="/api/classes/classdevice__new__allocator.html#typedef-difference_type">difference&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;struct <b><a href="/api/classes/structdevice__new__allocator_1_1rebind.html">rebind</a></b>;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classdevice__new__allocator.html#function-device_new_allocator">device&#95;new&#95;allocator</a></b>();</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classdevice__new__allocator.html#function-~device_new_allocator">~device&#95;new&#95;allocator</a></b>();</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classdevice__new__allocator.html#function-device_new_allocator">device&#95;new&#95;allocator</a></b>(<a href="/api/classes/classdevice__new__allocator.html">device_new_allocator</a> const &);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classdevice__new__allocator.html#function-device_new_allocator">device&#95;new&#95;allocator</a></b>(<a href="/api/classes/classdevice__new__allocator.html">device_new_allocator</a>< U > const &);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ <a href="/api/classes/classdevice__new__allocator.html#typedef-pointer">pointer</a> </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classdevice__new__allocator.html#function-address">address</a></b>(<a href="/api/classes/classdevice__new__allocator.html#typedef-reference">reference</a> r);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ <a href="/api/classes/classdevice__new__allocator.html#typedef-const_pointer">const_pointer</a> </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classdevice__new__allocator.html#function-address">address</a></b>(<a href="/api/classes/classdevice__new__allocator.html#typedef-const_reference">const_reference</a> r);</span>
<br>
<span>&nbsp;&nbsp;__host__ <a href="/api/classes/classdevice__new__allocator.html#typedef-pointer">pointer</a> </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classdevice__new__allocator.html#function-allocate">allocate</a></b>(<a href="/api/classes/classdevice__new__allocator.html#typedef-size_type">size_type</a> cnt,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;<a href="/api/classes/classdevice__new__allocator.html#typedef-const_pointer">const_pointer</a> = const&#95;pointer(static&#95;cast&lt; T &#42; &gt;(0)));</span>
<br>
<span>&nbsp;&nbsp;__host__ void </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classdevice__new__allocator.html#function-deallocate">deallocate</a></b>(<a href="/api/classes/classdevice__new__allocator.html#typedef-pointer">pointer</a> p,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;<a href="/api/classes/classdevice__new__allocator.html#typedef-size_type">size_type</a> cnt);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ <a href="/api/classes/classdevice__new__allocator.html#typedef-size_type">size_type</a> </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classdevice__new__allocator.html#function-max_size">max&#95;size</a></b>() const;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ bool </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classdevice__new__allocator.html#function-operator==">operator==</a></b>(<a href="/api/classes/classdevice__new__allocator.html">device_new_allocator</a> const &);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ bool </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classdevice__new__allocator.html#function-operator!=">operator!=</a></b>(<a href="/api/classes/classdevice__new__allocator.html">device_new_allocator</a> const & a);</span>
<span>};</span>
</code>

## Member Classes

<h3 id="struct-device_new_allocator::rebind">
<a href="/api/classes/structdevice__new__allocator_1_1rebind.html">Struct <code>device&#95;new&#95;allocator::device&#95;new&#95;allocator::rebind</code>
</a>
</h3>


## Member Types

<h3 id="typedef-value_type">
Typedef <code>device&#95;new&#95;allocator::value&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef T<b>value_type</b>;</span></code>
Type of element allocated, <code>T</code>. 

<h3 id="typedef-pointer">
Typedef <code>device&#95;new&#95;allocator::pointer</code>
</h3>

<code class="doxybook">
<span>typedef <a href="/api/classes/classdevice__ptr.html">device_ptr</a>< T ><b>pointer</b>;</span></code>
Pointer to allocation, <code>device&#95;ptr&lt;T&gt;</code>. 

<h3 id="typedef-const_pointer">
Typedef <code>device&#95;new&#95;allocator::const&#95;pointer</code>
</h3>

<code class="doxybook">
<span>typedef <a href="/api/classes/classdevice__ptr.html">device_ptr</a>< const T ><b>const_pointer</b>;</span></code>
<code>const</code> pointer to allocation, <code>device&#95;ptr&lt;const T&gt;</code>. 

<h3 id="typedef-reference">
Typedef <code>device&#95;new&#95;allocator::reference</code>
</h3>

<code class="doxybook">
<span>typedef <a href="/api/classes/classdevice__reference.html">device_reference</a>< T ><b>reference</b>;</span></code>
Reference to allocated element, <code>device&#95;reference&lt;T&gt;</code>. 

<h3 id="typedef-const_reference">
Typedef <code>device&#95;new&#95;allocator::const&#95;reference</code>
</h3>

<code class="doxybook">
<span>typedef <a href="/api/classes/classdevice__reference.html">device_reference</a>< const T ><b>const_reference</b>;</span></code>
<code>const</code> reference to allocated element, <code>device&#95;reference&lt;const T&gt;</code>. 

<h3 id="typedef-size_type">
Typedef <code>device&#95;new&#95;allocator::size&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef std::size_t<b>size_type</b>;</span></code>
Type of allocation size, <code>std::size&#95;t</code>. 

<h3 id="typedef-difference_type">
Typedef <code>device&#95;new&#95;allocator::difference&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef pointer::difference_type<b>difference_type</b>;</span></code>
Type of allocation difference, <code>pointer::difference&#95;type</code>. 


## Member Functions

<h3 id="function-device_new_allocator">
Function <code>device&#95;new&#95;allocator::&gt;::device&#95;new&#95;allocator</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>device_new_allocator</b>();</span></code>
No-argument constructor has no effect. 

<h3 id="function-~device_new_allocator">
Function <code>device&#95;new&#95;allocator::&gt;::~device&#95;new&#95;allocator</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>~device_new_allocator</b>();</span></code>
No-argument destructor has no effect. 

<h3 id="function-device_new_allocator">
Function <code>device&#95;new&#95;allocator::&gt;::device&#95;new&#95;allocator</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>device_new_allocator</b>(<a href="/api/classes/classdevice__new__allocator.html">device_new_allocator</a> const &);</span></code>
Copy constructor has no effect. 

<h3 id="function-device_new_allocator">
Function <code>device&#95;new&#95;allocator::&gt;::device&#95;new&#95;allocator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ </span><span><b>device_new_allocator</b>(<a href="/api/classes/classdevice__new__allocator.html">device_new_allocator</a>< U > const &);</span></code>
Constructor from other <code><a href="/api/classes/classdevice__malloc__allocator.html">device&#95;malloc&#95;allocator</a></code> has no effect. 

<h3 id="function-address">
Function <code>device&#95;new&#95;allocator::&gt;::address</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ <a href="/api/classes/classdevice__new__allocator.html#typedef-pointer">pointer</a> </span><span><b>address</b>(<a href="/api/classes/classdevice__new__allocator.html#typedef-reference">reference</a> r);</span></code>
Returns the address of an allocated object. 

**Returns**:
<code>&r</code>. 

<h3 id="function-address">
Function <code>device&#95;new&#95;allocator::&gt;::address</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ <a href="/api/classes/classdevice__new__allocator.html#typedef-const_pointer">const_pointer</a> </span><span><b>address</b>(<a href="/api/classes/classdevice__new__allocator.html#typedef-const_reference">const_reference</a> r);</span></code>
Returns the address an allocated object. 

**Returns**:
<code>&r</code>. 

<h3 id="function-allocate">
Function <code>device&#95;new&#95;allocator::&gt;::allocate</code>
</h3>

<code class="doxybook">
<span>__host__ <a href="/api/classes/classdevice__new__allocator.html#typedef-pointer">pointer</a> </span><span><b>allocate</b>(<a href="/api/classes/classdevice__new__allocator.html#typedef-size_type">size_type</a> cnt,</span>
<span>&nbsp;&nbsp;<a href="/api/classes/classdevice__new__allocator.html#typedef-const_pointer">const_pointer</a> = const&#95;pointer(static&#95;cast&lt; T &#42; &gt;(0)));</span></code>
Allocates storage for <code>cnt</code> objects. 

**Note**:
Memory allocated by this function must be deallocated with <code>deallocate</code>. 

**Function Parameters**:
**`cnt`**: The number of objects to allocate. 

**Returns**:
A <code>pointer</code> to uninitialized storage for <code>cnt</code> objects. 

<h3 id="function-deallocate">
Function <code>device&#95;new&#95;allocator::&gt;::deallocate</code>
</h3>

<code class="doxybook">
<span>__host__ void </span><span><b>deallocate</b>(<a href="/api/classes/classdevice__new__allocator.html#typedef-pointer">pointer</a> p,</span>
<span>&nbsp;&nbsp;<a href="/api/classes/classdevice__new__allocator.html#typedef-size_type">size_type</a> cnt);</span></code>
Deallocates storage for objects allocated with <code>allocate</code>. 

**Note**:
Memory deallocated by this function must previously have been allocated with <code>allocate</code>. 

**Function Parameters**:
* **`p`** A <code>pointer</code> to the storage to deallocate. 
* **`cnt`** The size of the previous allocation. 

<h3 id="function-max_size">
Function <code>device&#95;new&#95;allocator::&gt;::max&#95;size</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ <a href="/api/classes/classdevice__new__allocator.html#typedef-size_type">size_type</a> </span><span><b>max_size</b>() const;</span></code>
Returns the largest value <code>n</code> for which <code>allocate(n)</code> might succeed. 

**Returns**:
The largest value <code>n</code> for which <code>allocate(n)</code> might succeed. 

<h3 id="function-operator==">
Function <code>device&#95;new&#95;allocator::&gt;::operator==</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ bool </span><span><b>operator==</b>(<a href="/api/classes/classdevice__new__allocator.html">device_new_allocator</a> const &);</span></code>
Compares against another <code><a href="/api/classes/classdevice__malloc__allocator.html">device&#95;malloc&#95;allocator</a></code> for equality. 

**Returns**:
<code>true</code>

<h3 id="function-operator!=">
Function <code>device&#95;new&#95;allocator::&gt;::operator!=</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ bool </span><span><b>operator!=</b>(<a href="/api/classes/classdevice__new__allocator.html">device_new_allocator</a> const & a);</span></code>
Compares against another <code><a href="/api/classes/classdevice__malloc__allocator.html">device&#95;malloc&#95;allocator</a></code> for inequality. 

**Returns**:
<code>false</code>


