---
title: Containers
parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Containers

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename Alloc = thrust::device&#95;allocator&lt;T&gt;&gt;</span>
<span>class <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">thrust::device&#95;vector</a></b>;</span>
<br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename Alloc&gt;</span>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/groups/group__containers.html#function-swap">thrust::swap</a></b>(device_vector< T, Alloc > & a,</span>
<span>&nbsp;&nbsp;device_vector< T, Alloc > & b);</span>
</code>

## Member Classes

<h3 id="class-thrustdevice-vector">
<a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">Class <code>thrust::device&#95;vector</code>
</a>
</h3>

**Inherits From**:
`detail::vector_base< T, thrust::device_allocator< T > >`


## Functions

<h3 id="function-swap">
Function <code>thrust::swap</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename Alloc&gt;</span>
<span>void </span><span><b>swap</b>(device_vector< T, Alloc > & a,</span>
<span>&nbsp;&nbsp;device_vector< T, Alloc > & b);</span></code>
Exchanges the values of two vectors. <code>x</code> The first <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device&#95;vector</a></code> of interest. <code>y</code> The second <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">device&#95;vector</a></code> of interest. 


