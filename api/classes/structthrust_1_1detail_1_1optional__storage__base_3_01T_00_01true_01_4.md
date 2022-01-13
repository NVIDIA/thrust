---
title: thrust::detail::optional_storage_base< T, true >
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `thrust::detail::optional_storage_base< T, true >`

<code class="doxybook">
<span>template &lt;class T&gt;</span>
<span>struct thrust::detail::optional&#95;storage&#95;base&lt; T, true &gt; {</span>
<span>public:</span><span>&nbsp;&nbsp;struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__storage__base_3_01t_00_01true_01_4_1_1dummy.html">dummy</a></b>;</span>
<br>
<span>&nbsp;&nbsp;dummy <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__storage__base_3_01t_00_01true_01_4.html#variable-m-dummy">m&#95;dummy</a></b>;</span>
<br>
<span>&nbsp;&nbsp;T <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__storage__base_3_01t_00_01true_01_4.html#variable-m-value">m&#95;value</a></b>;</span>
<br>
<span>&nbsp;&nbsp;union thrust::detail::optional_storage_base< T, true >::@2 <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__storage__base_3_01t_00_01true_01_4.html#variable-@3">@3</a></b>;</span>
<br>
<span>&nbsp;&nbsp;bool <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__storage__base_3_01t_00_01true_01_4.html#variable-m-has-value">m&#95;has&#95;value</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ constexpr </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__storage__base_3_01t_00_01true_01_4.html#function-optional-storage-base">optional&#95;storage&#95;base</a></b>();</span>
<br>
<span>&nbsp;&nbsp;template &lt;class... U&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ constexpr </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__storage__base_3_01t_00_01true_01_4.html#function-optional-storage-base">optional&#95;storage&#95;base</a></b>(<a href="{{ site.baseurl }}/api/classes/structthrust_1_1in__place__t.html">in_place_t</a>,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;U &&... u);</span>
<span>};</span>
</code>

## Member Classes

<h3 id="struct-thrustdetailoptional-storage-base<-t,-true->dummy">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__storage__base_3_01t_00_01true_01_4_1_1dummy.html">Struct <code>thrust::detail::optional&#95;storage&#95;base&lt; T, true &gt;::dummy</code>
</a>
</h3>


## Member Variables

<h3 id="variable-m-dummy">
Variable <code>thrust::detail::optional&#95;storage&#95;base&lt; T, true &gt;::m&#95;dummy</code>
</h3>

<code class="doxybook">
<span>dummy <b>m_dummy</b>;</span></code>
<h3 id="variable-m-value">
Variable <code>thrust::detail::optional&#95;storage&#95;base&lt; T, true &gt;::m&#95;value</code>
</h3>

<code class="doxybook">
<span>T <b>m_value</b>;</span></code>
<h3 id="variable-@3">
Variable <code>thrust::detail::optional&#95;storage&#95;base&lt; T, true &gt;::@3</code>
</h3>

<code class="doxybook">
<span>union thrust::detail::optional_storage_base< T, true >::@2 <b>@3</b>;</span></code>
<h3 id="variable-m-has-value">
Variable <code>thrust::detail::optional&#95;storage&#95;base&lt; T, true &gt;::m&#95;has&#95;value</code>
</h3>

<code class="doxybook">
<span>bool <b>m_has_value</b> = false;</span></code>

## Member Functions

<h3 id="function-optional-storage-base">
Function <code>thrust::detail::optional&#95;storage&#95;base&lt; T, true &gt;::optional&#95;storage&#95;base</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ __device__ constexpr </span><span><b>optional_storage_base</b>();</span></code>
<h3 id="function-optional-storage-base">
Function <code>thrust::detail::optional&#95;storage&#95;base&lt; T, true &gt;::optional&#95;storage&#95;base</code>
</h3>

<code class="doxybook">
<span>template &lt;class... U&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ constexpr </span><span><b>optional_storage_base</b>(<a href="{{ site.baseurl }}/api/classes/structthrust_1_1in__place__t.html">in_place_t</a>,</span>
<span>&nbsp;&nbsp;U &&... u);</span></code>

