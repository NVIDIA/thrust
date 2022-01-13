---
title: thrust::tagged_deleter
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `thrust::tagged_deleter`

**Inherits From**:
`Lambda`

<code class="doxybook">
<span>template &lt;typename Pointer,</span>
<span>&nbsp;&nbsp;typename Lambda&gt;</span>
<span>struct thrust::tagged&#95;deleter {</span>
<span>public:</span><span>&nbsp;&nbsp;using <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1tagged__deleter.html#using-pointer">pointer</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1tagged__deleter.html#function-tagged-deleter">tagged&#95;deleter</a></b>(Lambda && l);</span>
<span>};</span>
</code>

## Member Types

<h3 id="using-pointer">
Type Alias <code>thrust::tagged&#95;deleter::pointer</code>
</h3>

<code class="doxybook">
<span>using <b>pointer</b> = Pointer;</span></code>

## Member Functions

<h3 id="function-tagged-deleter">
Function <code>thrust::tagged&#95;deleter::tagged&#95;deleter</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>tagged_deleter</b>(Lambda && l);</span></code>

