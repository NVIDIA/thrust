---
title: system::cuda
nav_exclude: true
has_children: true
has_toc: false
---

# Namespace `system::cuda`

<code class="doxybook">
<span>namespace system::cuda {</span>
<br>
<span>namespace <b><a href="/thrust/api/namespaces/namespacesystem_1_1cuda_1_1errc.html">system::cuda::errc</a></b> { <i>…</i> }</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structsystem_1_1cuda_1_1ready__future.html">ready&#95;future</a></b>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structsystem_1_1cuda_1_1unique__eager__future.html">unique&#95;eager&#95;future</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="/thrust/api/namespaces/namespacesystem_1_1cuda.html#typedef-memory_resource">memory&#95;resource</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="/thrust/api/namespaces/namespacesystem_1_1cuda.html#typedef-universal_memory_resource">universal&#95;memory&#95;resource</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="/thrust/api/namespaces/namespacesystem_1_1cuda.html#typedef-universal_host_pinned_memory_resource">universal&#95;host&#95;pinned&#95;memory&#95;resource</a></b>;</span>
<br>
<span>template &lt;typename... Events&gt;</span>
<span>__host__ unique_eager_event </span><span><b><a href="/thrust/api/namespaces/namespacesystem_1_1cuda.html#function-when_all">when&#95;all</a></b>(Events &&... evs);</span>
<span>} /* namespace system::cuda */</span>
</code>

## Member Classes

<h3 id="struct-system::cuda::ready_future">
<a href="/thrust/api/classes/structsystem_1_1cuda_1_1ready__future.html">Struct <code>system::cuda::ready&#95;future</code>
</a>
</h3>

<h3 id="struct-system::cuda::unique_eager_future">
<a href="/thrust/api/classes/structsystem_1_1cuda_1_1unique__eager__future.html">Struct <code>system::cuda::unique&#95;eager&#95;future</code>
</a>
</h3>


## Types

<h3 id="typedef-memory_resource">
Typedef <code>memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>typedef detail::device_memory_resource<b>memory_resource</b>;</span></code>
The memory resource for the CUDA system. Uses <code>cudaMalloc</code> and wraps the result with <code>cuda::pointer</code>. 

<h3 id="typedef-universal_memory_resource">
Typedef <code>universal&#95;memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>typedef detail::managed_memory_resource<b>universal_memory_resource</b>;</span></code>
The universal memory resource for the CUDA system. Uses <code>cudaMallocManaged</code> and wraps the result with <code>cuda::universal&#95;pointer</code>. 

<h3 id="typedef-universal_host_pinned_memory_resource">
Typedef <code>universal&#95;host&#95;pinned&#95;memory&#95;resource</code>
</h3>

<code class="doxybook">
<span>typedef detail::pinned_memory_resource<b>universal_host_pinned_memory_resource</b>;</span></code>
The host pinned memory resource for the CUDA system. Uses <code>cudaMallocHost</code> and wraps the result with <code>cuda::universal&#95;pointer</code>. 


## Functions

<h3 id="function-when_all">
Function <code>system::cuda::when&#95;all</code>
</h3>

<code class="doxybook">
<span>template &lt;typename... Events&gt;</span>
<span>__host__ unique_eager_event </span><span><b>when_all</b>(Events &&... evs);</span></code>

