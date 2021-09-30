---
title: detail::optional_operations_base
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `detail::optional_operations_base`

**Inherits From**:
`detail::optional_storage_base< T >`

**Inherited By**:
* [`detail::optional_copy_base< T, bool >`](/thrust/api/classes/structdetail_1_1optional__copy__base.html)
* [`detail::optional_copy_base< T, false >`](/thrust/api/classes/structdetail_1_1optional__copy__base_3_01t_00_01false_01_4.html)
* [`detail::optional_copy_base< T >`](/thrust/api/classes/structdetail_1_1optional__copy__base.html)

<code class="doxybook">
<span>template &lt;class T&gt;</span>
<span>struct detail::optional&#95;operations&#95;base {</span>
<span>public:</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ void </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/structdetail_1_1optional__operations__base.html#function-hard_reset">hard&#95;reset</a></b>();</span>
<br>
<span>&nbsp;&nbsp;template &lt;class... Args&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ void </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/structdetail_1_1optional__operations__base.html#function-construct">construct</a></b>(Args &&... args);</span>
<br>
<span>&nbsp;&nbsp;template &lt;class Opt&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ void </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/structdetail_1_1optional__operations__base.html#function-assign">assign</a></b>(Opt && rhs);</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ bool </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/structdetail_1_1optional__operations__base.html#function-has_value">has&#95;value</a></b>() const;</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ T & </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/structdetail_1_1optional__operations__base.html#function-get">get</a></b>();</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr const __device__ T & </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/structdetail_1_1optional__operations__base.html#function-get">get</a></b>() const;</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ T && </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/structdetail_1_1optional__operations__base.html#function-get">get</a></b>();</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr const __device__ T && </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/structdetail_1_1optional__operations__base.html#function-get">get</a></b>() const;</span>
<span>};</span>
</code>

## Member Functions

<h3 id="function-hard_reset">
Function <code>detail::optional&#95;operations&#95;base::&gt;::hard&#95;reset</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ __device__ void </span><span><b>hard_reset</b>();</span></code>
<h3 id="function-construct">
Function <code>detail::optional&#95;operations&#95;base::&gt;::construct</code>
</h3>

<code class="doxybook">
<span>template &lt;class... Args&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ void </span><span><b>construct</b>(Args &&... args);</span></code>
<h3 id="function-assign">
Function <code>detail::optional&#95;operations&#95;base::&gt;::assign</code>
</h3>

<code class="doxybook">
<span>template &lt;class Opt&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ void </span><span><b>assign</b>(Opt && rhs);</span></code>
<h3 id="function-has_value">
Function <code>detail::optional&#95;operations&#95;base::&gt;::has&#95;value</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ __device__ bool </span><span><b>has_value</b>() const;</span></code>
<h3 id="function-get">
Function <code>detail::optional&#95;operations&#95;base::&gt;::get</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ T & </span><span><b>get</b>();</span></code>
<h3 id="function-get">
Function <code>detail::optional&#95;operations&#95;base::&gt;::get</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr const __device__ T & </span><span><b>get</b>() const;</span></code>
<h3 id="function-get">
Function <code>detail::optional&#95;operations&#95;base::&gt;::get</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ T && </span><span><b>get</b>();</span></code>
<h3 id="function-get">
Function <code>detail::optional&#95;operations&#95;base::&gt;::get</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr const __device__ T && </span><span><b>get</b>() const;</span></code>

