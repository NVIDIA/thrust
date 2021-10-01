---
title: device_allocator
summary: An allocator which creates new elements in memory accessible by devices. 
parent: Allocators
grand_parent: Memory Management
nav_exclude: true
has_children: true
has_toc: false
---

# Class `device_allocator`

An allocator which creates new elements in memory accessible by devices. 

**Inherits From**:
`thrust::mr::stateless_resource_allocator< T, device_ptr_memory_resource< device_memory_resource > >`

**See**:
<a href="https://en.cppreference.com/w/cpp/named_req/Allocator">https://en.cppreference.com/w/cpp/named_req/Allocator</a>

<code class="doxybook">
<span>#include <thrust/device_allocator.h></span><br>
<span>template &lt;typename T&gt;</span>
<span>class device&#95;allocator {</span>
<span>public:</span><span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;struct <b><a href="/thrust/api/classes/structdevice__allocator_1_1rebind.html">rebind</a></b>;</span>
<br>
<span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classdevice__allocator.html#function-device_allocator">device&#95;allocator</a></b>();</span>
<br>
<span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classdevice__allocator.html#function-device_allocator">device&#95;allocator</a></b>(const <a href="/thrust/api/classes/classdevice__allocator.html">device_allocator</a> & other);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classdevice__allocator.html#function-device_allocator">device&#95;allocator</a></b>(const <a href="/thrust/api/classes/classdevice__allocator.html">device_allocator</a>< U > & other);</span>
<br>
<span>&nbsp;&nbsp;<a href="/thrust/api/classes/classdevice__allocator.html">device_allocator</a> & </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classdevice__allocator.html#function-operator=">operator=</a></b>(const <a href="/thrust/api/classes/classdevice__allocator.html">device_allocator</a> &) = default;</span>
<br>
<span>&nbsp;&nbsp;__host__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classdevice__allocator.html#function-~device_allocator">~device&#95;allocator</a></b>();</span>
<span>};</span>
</code>

## Member Classes

<h3 id="struct-device_allocator::rebind">
<a href="/thrust/api/classes/structdevice__allocator_1_1rebind.html">Struct <code>device&#95;allocator::device&#95;allocator::rebind</code>
</a>
</h3>


## Member Functions

<h3 id="function-device_allocator">
Function <code>device&#95;allocator::&gt;::device&#95;allocator</code>
</h3>

<code class="doxybook">
<span>__host__ </span><span><b>device_allocator</b>();</span></code>
Default constructor has no effect. 

<h3 id="function-device_allocator">
Function <code>device&#95;allocator::&gt;::device&#95;allocator</code>
</h3>

<code class="doxybook">
<span>__host__ </span><span><b>device_allocator</b>(const <a href="/thrust/api/classes/classdevice__allocator.html">device_allocator</a> & other);</span></code>
Copy constructor has no effect. 

<h3 id="function-device_allocator">
Function <code>device&#95;allocator::&gt;::device&#95;allocator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>__host__ </span><span><b>device_allocator</b>(const <a href="/thrust/api/classes/classdevice__allocator.html">device_allocator</a>< U > & other);</span></code>
Constructor from other <code><a href="/thrust/api/classes/classdevice__allocator.html">device&#95;allocator</a></code> has no effect. 

<h3 id="function-operator=">
Function <code>device&#95;allocator::&gt;::operator=</code>
</h3>

<code class="doxybook">
<span><a href="/thrust/api/classes/classdevice__allocator.html">device_allocator</a> & </span><span><b>operator=</b>(const <a href="/thrust/api/classes/classdevice__allocator.html">device_allocator</a> &) = default;</span></code>
<h3 id="function-~device_allocator">
Function <code>device&#95;allocator::&gt;::~device&#95;allocator</code>
</h3>

<code class="doxybook">
<span>__host__ </span><span><b>~device_allocator</b>();</span></code>
Destructor has no effect. 


