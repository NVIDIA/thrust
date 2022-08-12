---
title: thrust::detail::zip_detail
nav_exclude: true
has_children: true
has_toc: false
---

# Namespace `thrust::detail::zip_detail`

<code class="doxybook">
<span>namespace thrust::detail::zip&#95;detail {</span>
<br>
<span>template &lt;typename Function,</span>
<span>&nbsp;&nbsp;typename Tuple,</span>
<span>&nbsp;&nbsp;std::size_t... Is&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ decltype(auto) </span><span><b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail_1_1zip__detail.html#function-apply-impl">apply&#95;impl</a></b>(Function && func,</span>
<span>&nbsp;&nbsp;Tuple && args,</span>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/groups/group__type__traits.html#using-index-sequence">index_sequence</a>< Is... >);</span>
<br>
<span>template &lt;typename Function,</span>
<span>&nbsp;&nbsp;typename Tuple&gt;</span>
<span>__host__ __device__ decltype(auto) </span><span><b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail_1_1zip__detail.html#function-apply">apply</a></b>(Function && func,</span>
<span>&nbsp;&nbsp;Tuple && args);</span>
<span>} /* namespace thrust::detail::zip&#95;detail */</span>
</code>

## Functions

<h3 id="function-apply-impl">
Function <code>thrust::detail::zip&#95;detail::apply&#95;impl</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Function,</span>
<span>&nbsp;&nbsp;typename Tuple,</span>
<span>&nbsp;&nbsp;std::size_t... Is&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ decltype(auto) </span><span><b>apply_impl</b>(Function && func,</span>
<span>&nbsp;&nbsp;Tuple && args,</span>
<span>&nbsp;&nbsp;<a href="{{ site.baseurl }}/api/groups/group__type__traits.html#using-index-sequence">index_sequence</a>< Is... >);</span></code>
<h3 id="function-apply">
Function <code>thrust::detail::zip&#95;detail::apply</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Function,</span>
<span>&nbsp;&nbsp;typename Tuple&gt;</span>
<span>__host__ __device__ decltype(auto) </span><span><b>apply</b>(Function && func,</span>
<span>&nbsp;&nbsp;Tuple && args);</span></code>

