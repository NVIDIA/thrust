---
title: thrust::maximum< void >
parent: Generalized Identity Operations
grand_parent: Predefined Function Objects
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `thrust::maximum< void >`

<code class="doxybook">
<span>struct thrust::maximum&lt; void &gt; {</span>
<span>public:</span><span>&nbsp;&nbsp;using <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1maximum_3_01void_01_4.html#using-is-transparent">is&#95;transparent</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename T1,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typename T2&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ constexpr auto </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1maximum_3_01void_01_4.html#function-operator()">operator()</a></b>(T1 && t1,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;T2 && t2) const;</span>
<span>};</span>
</code>

## Member Types

<h3 id="using-is-transparent">
Type Alias <code>thrust::maximum&lt; void &gt;::is&#95;transparent</code>
</h3>

<code class="doxybook">
<span>using <b>is_transparent</b> = void;</span></code>

## Member Functions

<h3 id="function-operator()">
Function <code>thrust::maximum&lt; void &gt;::operator()</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T1,</span>
<span>&nbsp;&nbsp;typename T2&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ constexpr auto </span><span><b>operator()</b>(T1 && t1,</span>
<span>&nbsp;&nbsp;T2 && t2) const;</span></code>

