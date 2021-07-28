---
title: detail::zip_detail
nav_exclude: true
has_children: true
has_toc: false
---

# Namespace `detail::zip_detail`

<code class="doxybook">
<span>namespace detail::zip&#95;detail {</span>
<br>
<span>template &lt;typename Function,</span>
<span>&nbsp;&nbsp;typename Tuple,</span>
<span>&nbsp;&nbsp;std::size_t... Is&gt;</span>
<span>__host__decltype(auto) __device__ </span><span><b><a href="/api/namespaces/namespacedetail_1_1zip__detail.html#function-apply_impl">apply&#95;impl</a></b>(Function && func,</span>
<span>&nbsp;&nbsp;Tuple && args,</span>
<span>&nbsp;&nbsp;<a href="/api/groups/group__type__traits.html#using-index_sequence">index_sequence</a>< Is... >);</span>
<br>
<span>template &lt;typename Function,</span>
<span>&nbsp;&nbsp;typename Tuple&gt;</span>
<span>__host__decltype(auto) __device__ </span><span><b><a href="/api/namespaces/namespacedetail_1_1zip__detail.html#function-apply">apply</a></b>(Function && func,</span>
<span>&nbsp;&nbsp;Tuple && args);</span>
<span>} /* namespace detail::zip&#95;detail */</span>
</code>

## Functions

<h3 id="function-apply_impl">
Function <code>detail::zip&#95;detail::apply&#95;impl</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Function,</span>
<span>&nbsp;&nbsp;typename Tuple,</span>
<span>&nbsp;&nbsp;std::size_t... Is&gt;</span>
<span>__host__decltype(auto) __device__ </span><span><b>apply_impl</b>(Function && func,</span>
<span>&nbsp;&nbsp;Tuple && args,</span>
<span>&nbsp;&nbsp;<a href="/api/groups/group__type__traits.html#using-index_sequence">index_sequence</a>< Is... >);</span></code>
<h3 id="function-apply">
Function <code>detail::zip&#95;detail::apply</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Function,</span>
<span>&nbsp;&nbsp;typename Tuple&gt;</span>
<span>__host__decltype(auto) __device__ </span><span><b>apply</b>(Function && func,</span>
<span>&nbsp;&nbsp;Tuple && args);</span></code>

