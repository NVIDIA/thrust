---
title: thrust::system::cuda
summary: thrust::system::cuda is the namespace containing functionality for allocating, manipulating, and deallocating memory available to Thrust's CUDA backend system. The identifiers are provided in a separate namespace underneath thrust::system for import convenience but are also aliased in the top-level thrust::cuda namespace for easy access. 
parent: Systems
grand_parent: System
nav_exclude: true
has_children: true
has_toc: false
---

# Namespace `thrust::system::cuda`

<code class="doxybook">
<span>namespace thrust::system::cuda {</span>
<br>
<span>namespace <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1cuda_1_1errc.html">thrust::system::cuda::errc</a></b> { <i>…</i> }</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1system_1_1cuda_1_1ready__future.html">ready&#95;future</a></b>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1system_1_1cuda_1_1unique__eager__future.html">unique&#95;eager&#95;future</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1cuda.html#typedef-memory-resource">memory&#95;resource</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1cuda.html#typedef-universal-memory-resource">universal&#95;memory&#95;resource</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1cuda.html#typedef-universal-host-pinned-memory-resource">universal&#95;host&#95;pinned&#95;memory&#95;resource</a></b>;</span>
<br>
<span>template &lt;typename... Events&gt;</span>
<span>__host__ unique_eager_event </span><span><b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1system_1_1cuda.html#function-when-all">when&#95;all</a></b>(Events &&... evs);</span>
<span>} /* namespace thrust::system::cuda */</span>
</code>

## Member Classes

<h3 id="struct-thrustsystemcudaready-future">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1system_1_1cuda_1_1ready__future.html">Struct <code>thrust::system::cuda::ready&#95;future</code>
</a>
</h3>

<h3 id="struct-thrustsystemcudaunique-eager-future">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1system_1_1cuda_1_1unique__eager__future.html">Struct <code>thrust::system::cuda::unique&#95;eager&#95;future</code>
</a>
</h3>


## Types

<h3 id="typedef-memory-resource">
Typedef <code>thrust::system::cuda::memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>typedef detail::device_memory_resource<b>memory_resource</b>;</span></code>
The memory resource for the CUDA system. Uses <code>cudaMalloc</code> and wraps the result with <code>cuda::pointer</code>. 

<h3 id="typedef-universal-memory-resource">
Typedef <code>thrust::system::cuda::universal&#95;memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>typedef detail::managed_memory_resource<b>universal_memory_resource</b>;</span></code>
The universal memory resource for the CUDA system. Uses <code>cudaMallocManaged</code> and wraps the result with <code>cuda::universal&#95;pointer</code>. 

<h3 id="typedef-universal-host-pinned-memory-resource">
Typedef <code>thrust::system::cuda::universal&#95;host&#95;pinned&#95;memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>typedef detail::pinned_memory_resource<b>universal_host_pinned_memory_resource</b>;</span></code>
The host pinned memory resource for the CUDA system. Uses <code>cudaMallocHost</code> and wraps the result with <code>cuda::universal&#95;pointer</code>. 


## Functions

<h3 id="function-when-all">
Function <code>thrust::system::cuda::when&#95;all</code>
</h3>

<code class="doxybook">
<span>template &lt;typename... Events&gt;</span>
<span>__host__ unique_eager_event </span><span><b>when_all</b>(Events &&... evs);</span></code>

