---
title: detail::optional_storage_base
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `detail::optional_storage_base`

**Inherited By**:
[`detail::optional_operations_base< T >`](/api/classes/structdetail_1_1optional__operations__base.html)

<code class="doxybook">
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;bool = ::std::is&#95;trivially&#95;destructible&lt;T&gt;::value&gt;</span>
<span>struct detail::optional&#95;storage&#95;base {</span>
<span>public:</span><span>&nbsp;&nbsp;struct <b><a href="/api/classes/structdetail_1_1optional__storage__base_1_1dummy.html">dummy</a></b>;</span>
<br>
<span>&nbsp;&nbsp;dummy <b><a href="/api/classes/structdetail_1_1optional__storage__base.html#variable-m_dummy">m&#95;dummy</a></b>;</span>
<br>
<span>&nbsp;&nbsp;T <b><a href="/api/classes/structdetail_1_1optional__storage__base.html#variable-m_value">m&#95;value</a></b>;</span>
<br>
<span>&nbsp;&nbsp;union detail::optional_storage_base::@0 <b><a href="/api/classes/structdetail_1_1optional__storage__base.html#variable-@1">@1</a></b>;</span>
<br>
<span>&nbsp;&nbsp;bool <b><a href="/api/classes/structdetail_1_1optional__storage__base.html#variable-m_has_value">m&#95;has&#95;value</a></b>;</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ </span><span>&nbsp;&nbsp;<b><a href="/api/classes/structdetail_1_1optional__storage__base.html#function-optional_storage_base">optional&#95;storage&#95;base</a></b>();</span>
<br>
<span>&nbsp;&nbsp;template &lt;class... U&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ </span><span>&nbsp;&nbsp;<b><a href="/api/classes/structdetail_1_1optional__storage__base.html#function-optional_storage_base">optional&#95;storage&#95;base</a></b>(<a href="/api/classes/structin__place__t.html">in_place_t</a>,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;U &&... u);</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="/api/classes/structdetail_1_1optional__storage__base.html#function-~optional_storage_base">~optional&#95;storage&#95;base</a></b>();</span>
<span>};</span>
</code>

## Member Classes

<h3 id="struct-detail::optional_storage_base::dummy">
<a href="/api/classes/structdetail_1_1optional__storage__base_1_1dummy.html">Struct <code>detail::optional&#95;storage&#95;base::detail::optional&#95;storage&#95;base::dummy</code>
</a>
</h3>


## Member Variables

<h3 id="variable-m_dummy">
Variable <code>detail::optional&#95;storage&#95;base::detail::optional&#95;storage&#95;base::m&#95;dummy</code>
</h3>

<code class="doxybook">
<span>dummy <b>m_dummy</b>;</span></code>
<h3 id="variable-m_value">
Variable <code>detail::optional&#95;storage&#95;base::detail::optional&#95;storage&#95;base::m&#95;value</code>
</h3>

<code class="doxybook">
<span>T <b>m_value</b>;</span></code>
<h3 id="variable-@1">
Variable <code>detail::optional&#95;storage&#95;base::detail::optional&#95;storage&#95;base::@1</code>
</h3>

<code class="doxybook">
<span>union detail::optional_storage_base::@0 <b>@1</b>;</span></code>
<h3 id="variable-m_has_value">
Variable <code>detail::optional&#95;storage&#95;base::detail::optional&#95;storage&#95;base::m&#95;has&#95;value</code>
</h3>

<code class="doxybook">
<span>bool <b>m_has_value</b>;</span></code>

## Member Functions

<h3 id="function-optional_storage_base">
Function <code>detail::optional&#95;storage&#95;base::&gt;::optional&#95;storage&#95;base</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ </span><span><b>optional_storage_base</b>();</span></code>
<h3 id="function-optional_storage_base">
Function <code>detail::optional&#95;storage&#95;base::&gt;::optional&#95;storage&#95;base</code>
</h3>

<code class="doxybook">
<span>template &lt;class... U&gt;</span>
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ </span><span><b>optional_storage_base</b>(<a href="/api/classes/structin__place__t.html">in_place_t</a>,</span>
<span>&nbsp;&nbsp;U &&... u);</span></code>
<h3 id="function-~optional_storage_base">
Function <code>detail::optional&#95;storage&#95;base::&gt;::~optional&#95;storage&#95;base</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ __device__ </span><span><b>~optional_storage_base</b>();</span></code>

