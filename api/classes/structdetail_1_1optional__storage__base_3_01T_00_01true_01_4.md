---
title: detail::optional_storage_base< T, true >
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `detail::optional_storage_base< T, true >`

<code class="doxybook">
<span>template &lt;class T&gt;</span>
<span>struct detail::optional&#95;storage&#95;base&lt; T, true &gt; {</span>
<span>public:</span><span>&nbsp;&nbsp;struct <b><a href="/api/classes/structdetail_1_1optional__storage__base_3_01t_00_01true_01_4_1_1dummy.html">dummy</a></b>;</span>
<br>
<span>&nbsp;&nbsp;dummy <b><a href="/api/classes/structdetail_1_1optional__storage__base_3_01t_00_01true_01_4.html#variable-m_dummy">m&#95;dummy</a></b>;</span>
<br>
<span>&nbsp;&nbsp;T <b><a href="/api/classes/structdetail_1_1optional__storage__base_3_01t_00_01true_01_4.html#variable-m_value">m&#95;value</a></b>;</span>
<br>
<span>&nbsp;&nbsp;union detail::optional_storage_base< T, true >::@2 <b><a href="/api/classes/structdetail_1_1optional__storage__base_3_01t_00_01true_01_4.html#variable-@3">@3</a></b>;</span>
<br>
<span>&nbsp;&nbsp;bool <b><a href="/api/classes/structdetail_1_1optional__storage__base_3_01t_00_01true_01_4.html#variable-m_has_value">m&#95;has&#95;value</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ </span><span>&nbsp;&nbsp;<b><a href="/api/classes/structdetail_1_1optional__storage__base_3_01t_00_01true_01_4.html#function-optional_storage_base">optional&#95;storage&#95;base</a></b>();</span>
<br>
<span>&nbsp;&nbsp;template &lt;class... U&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ </span><span>&nbsp;&nbsp;<b><a href="/api/classes/structdetail_1_1optional__storage__base_3_01t_00_01true_01_4.html#function-optional_storage_base">optional&#95;storage&#95;base</a></b>(<a href="/api/classes/structin__place__t.html">in_place_t</a>,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;U &&... u);</span>
<span>};</span>
</code>

## Member Classes

<h3 id="struct-detail::optional_storage_base< T, true >::dummy">
<a href="/api/classes/structdetail_1_1optional__storage__base_3_01t_00_01true_01_4_1_1dummy.html">Struct <code>detail::optional&#95;storage&#95;base&lt; T, true &gt;::detail::optional&#95;storage&#95;base&lt; T, true &gt;::dummy</code>
</a>
</h3>


## Member Variables

<h3 id="variable-m_dummy">
Variable <code>detail::optional&#95;storage&#95;base&lt; T, true &gt;::detail::optional&#95;storage&#95;base&lt; T, true &gt;::m&#95;dummy</code>
</h3>

<code class="doxybook">
<span>dummy <b>m_dummy</b>;</span></code>
<h3 id="variable-m_value">
Variable <code>detail::optional&#95;storage&#95;base&lt; T, true &gt;::detail::optional&#95;storage&#95;base&lt; T, true &gt;::m&#95;value</code>
</h3>

<code class="doxybook">
<span>T <b>m_value</b>;</span></code>
<h3 id="variable-@3">
Variable <code>detail::optional&#95;storage&#95;base&lt; T, true &gt;::detail::optional&#95;storage&#95;base&lt; T, true &gt;::@3</code>
</h3>

<code class="doxybook">
<span>union detail::optional_storage_base< T, true >::@2 <b>@3</b>;</span></code>
<h3 id="variable-m_has_value">
Variable <code>detail::optional&#95;storage&#95;base&lt; T, true &gt;::detail::optional&#95;storage&#95;base&lt; T, true &gt;::m&#95;has&#95;value</code>
</h3>

<code class="doxybook">
<span>bool <b>m_has_value</b> = false;</span></code>

## Member Functions

<h3 id="function-optional_storage_base">
Function <code>detail::optional&#95;storage&#95;base&lt; T, true &gt;::&gt;::optional&#95;storage&#95;base</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ </span><span><b>optional_storage_base</b>();</span></code>
<h3 id="function-optional_storage_base">
Function <code>detail::optional&#95;storage&#95;base&lt; T, true &gt;::&gt;::optional&#95;storage&#95;base</code>
</h3>

<code class="doxybook">
<span>template &lt;class... U&gt;</span>
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ </span><span><b>optional_storage_base</b>(<a href="/api/classes/structin__place__t.html">in_place_t</a>,</span>
<span>&nbsp;&nbsp;U &&... u);</span></code>

