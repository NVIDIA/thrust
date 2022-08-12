---
title: thrust::device_allocator
summary: An allocator which creates new elements in memory accessible by devices. 
parent: Allocators
grand_parent: Memory Management
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::device_allocator`

An allocator which creates new elements in memory accessible by devices. 

**Inherits From**:
* `thrust::mr::stateless_resource_allocator< T, device_ptr_memory_resource< device_memory_resource > >`
* `thrust::mr::allocator< T, Upstream >`
* `thrust::mr::validator< MR >`

**See**:
<a href="https://en.cppreference.com/w/cpp/named_req/Allocator">https://en.cppreference.com/w/cpp/named_req/Allocator</a>

<code class="doxybook">
<span>#include <thrust/device_allocator.h></span><br>
<span>template &lt;typename T&gt;</span>
<span>class thrust::device&#95;allocator {</span>
<span>public:</span><span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">thrust::mr::allocator&lt; T, Upstream &gt;</a></b></code> */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-void-pointer">void&#95;pointer</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">thrust::mr::allocator&lt; T, Upstream &gt;</a></b></code> */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-value-type">value&#95;type</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">thrust::mr::allocator&lt; T, Upstream &gt;</a></b></code> */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-pointer">pointer</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">thrust::mr::allocator&lt; T, Upstream &gt;</a></b></code> */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-const-pointer">const&#95;pointer</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">thrust::mr::allocator&lt; T, Upstream &gt;</a></b></code> */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-reference">reference</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">thrust::mr::allocator&lt; T, Upstream &gt;</a></b></code> */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-const-reference">const&#95;reference</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">thrust::mr::allocator&lt; T, Upstream &gt;</a></b></code> */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-size-type">size&#95;type</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">thrust::mr::allocator&lt; T, Upstream &gt;</a></b></code> */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-difference-type">difference&#95;type</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">thrust::mr::allocator&lt; T, Upstream &gt;</a></b></code> */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-propagate-on-container-copy-assignment">propagate&#95;on&#95;container&#95;copy&#95;assignment</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">thrust::mr::allocator&lt; T, Upstream &gt;</a></b></code> */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-propagate-on-container-move-assignment">propagate&#95;on&#95;container&#95;move&#95;assignment</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">thrust::mr::allocator&lt; T, Upstream &gt;</a></b></code> */</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-propagate-on-container-swap">propagate&#95;on&#95;container&#95;swap</a></b>;</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1device__allocator_1_1rebind.html">rebind</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1stateless__resource__allocator.html">thrust::mr::stateless&#95;resource&#95;allocator&lt; T, device&#95;ptr&#95;memory&#95;resource&lt; device&#95;memory&#95;resource &gt; &gt;</a></b></code> */</span><span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1stateless__resource__allocator_1_1rebind.html">rebind</a></b>;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">thrust::mr::allocator&lt; T, Upstream &gt;</a></b></code> */</span><span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1allocator_1_1rebind.html">rebind</a></b>;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__allocator.html#function-device-allocator">device&#95;allocator</a></b>();</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__allocator.html#function-device-allocator">device&#95;allocator</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__allocator.html">device_allocator</a> & other);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__allocator.html#function-device-allocator">device&#95;allocator</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__allocator.html">device_allocator</a>< U > & other);</span>
<br>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__allocator.html">device_allocator</a> & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__allocator.html#function-operator=">operator=</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__allocator.html">device_allocator</a> &) = default;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__allocator.html#function-~device-allocator">~device&#95;allocator</a></b>();</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1stateless__resource__allocator.html">thrust::mr::stateless&#95;resource&#95;allocator&lt; T, device&#95;ptr&#95;memory&#95;resource&lt; device&#95;memory&#95;resource &gt; &gt;</a></b></code> */</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1stateless__resource__allocator.html#function-stateless-resource-allocator">stateless&#95;resource&#95;allocator</a></b>();</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1stateless__resource__allocator.html">thrust::mr::stateless&#95;resource&#95;allocator&lt; T, device&#95;ptr&#95;memory&#95;resource&lt; device&#95;memory&#95;resource &gt; &gt;</a></b></code> */</span><span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1stateless__resource__allocator.html#function-stateless-resource-allocator">stateless&#95;resource&#95;allocator</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1stateless__resource__allocator.html">stateless_resource_allocator</a> & other);</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1stateless__resource__allocator.html">thrust::mr::stateless&#95;resource&#95;allocator&lt; T, device&#95;ptr&#95;memory&#95;resource&lt; device&#95;memory&#95;resource &gt; &gt;</a></b></code> */</span><span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1stateless__resource__allocator.html#function-stateless-resource-allocator">stateless&#95;resource&#95;allocator</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1stateless__resource__allocator.html">stateless_resource_allocator</a>< U, Upstream > & other);</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1stateless__resource__allocator.html">thrust::mr::stateless&#95;resource&#95;allocator&lt; T, device&#95;ptr&#95;memory&#95;resource&lt; device&#95;memory&#95;resource &gt; &gt;</a></b></code> */</span><span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1stateless__resource__allocator.html#function-~stateless-resource-allocator">~stateless&#95;resource&#95;allocator</a></b>();</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">thrust::mr::allocator&lt; T, Upstream &gt;</a></b></code> */</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-size-type">size_type</a> </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#function-max-size">max&#95;size</a></b>() const;</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">thrust::mr::allocator&lt; T, Upstream &gt;</a></b></code> */</span><span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#function-allocator">allocator</a></b>(MR * resource);</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">thrust::mr::allocator&lt; T, Upstream &gt;</a></b></code> */</span><span>&nbsp;&nbsp;template &lt;typename U&gt;</span>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#function-allocator">allocator</a></b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">allocator</a>< U, MR > & other);</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">thrust::mr::allocator&lt; T, Upstream &gt;</a></b></code> */</span><span>&nbsp;&nbsp;__host__ <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-pointer">pointer</a> </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#function-allocate">allocate</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-size-type">size_type</a> n);</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">thrust::mr::allocator&lt; T, Upstream &gt;</a></b></code> */</span><span>&nbsp;&nbsp;__host__ void </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#function-deallocate">deallocate</a></b>(<a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-pointer">pointer</a> p,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#typedef-size-type">size_type</a> n);</span>
<br>
<span class="doxybook-comment"><code>&nbsp;&nbsp;</code>
/* Inherited from <code><b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">thrust::mr::allocator&lt; T, Upstream &gt;</a></b></code> */</span><span>&nbsp;&nbsp;__host__ __device__ MR * </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html#function-resource">resource</a></b>() const;</span>
<span>};</span>
</code>

## Member Classes

<h3 id="struct-thrustdevice-allocatorrebind">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1device__allocator_1_1rebind.html">Struct <code>thrust::device&#95;allocator::rebind</code>
</a>
</h3>


## Member Functions

<h3 id="function-device-allocator">
Function <code>thrust::device&#95;allocator::device&#95;allocator</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>device_allocator</b>();</span></code>
Default constructor has no effect. 

<h3 id="function-device-allocator">
Function <code>thrust::device&#95;allocator::device&#95;allocator</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>device_allocator</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__allocator.html">device_allocator</a> & other);</span></code>
Copy constructor has no effect. 

<h3 id="function-device-allocator">
Function <code>thrust::device&#95;allocator::device&#95;allocator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename U&gt;</span>
<span>__host__ __device__ </span><span><b>device_allocator</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__allocator.html">device_allocator</a>< U > & other);</span></code>
Constructor from other <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__allocator.html">device&#95;allocator</a></code> has no effect. 

<h3 id="function-operator=">
Function <code>thrust::device&#95;allocator::operator=</code>
</h3>

<code class="doxybook">
<span><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__allocator.html">device_allocator</a> & </span><span><b>operator=</b>(const <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__allocator.html">device_allocator</a> &) = default;</span></code>
<h3 id="function-~device-allocator">
Function <code>thrust::device&#95;allocator::~device&#95;allocator</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>~device_allocator</b>();</span></code>
Destructor has no effect. 


