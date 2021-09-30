---
title: greater< void >
parent: Comparison Operations
grand_parent: Predefined Function Objects
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `greater< void >`

<code class="doxybook">
<span>struct greater&lt; void &gt; {</span>
<span>public:</span><span>&nbsp;&nbsp;using <b><a href="/thrust/api/classes/structgreater_3_01void_01_4.html#using-is_transparent">is&#95;transparent</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename T1,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typename T2&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ auto </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/structgreater_3_01void_01_4.html#function-operator()">operator()</a></b>(T1 && t1,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;T2 && t2) const;</span>
<span>};</span>
</code>

## Member Types

<h3 id="using-is_transparent">
Type Alias <code>greater&lt; void &gt;::is&#95;transparent</code>
</h3>

<code class="doxybook">
<span>using <b>is_transparent</b> = void;</span></code>

## Member Functions

<h3 id="function-operator()">
Function <code>greater&lt; void &gt;::&gt;::operator()</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T1,</span>
<span>&nbsp;&nbsp;typename T2&gt;</span>
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ auto </span><span><b>operator()</b>(T1 && t1,</span>
<span>&nbsp;&nbsp;T2 && t2) const;</span></code>

