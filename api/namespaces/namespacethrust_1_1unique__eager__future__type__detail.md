---
title: thrust::unique_eager_future_type_detail
nav_exclude: true
has_children: true
has_toc: false
---

# Namespace `thrust::unique_eager_future_type_detail`

<code class="doxybook">
<span>namespace thrust::unique&#95;eager&#95;future&#95;type&#95;detail {</span>
<br>
<span>template &lt;typename System,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1unique__eager__future__type__detail.html#using-select">select</a></b> = <i>see below</i>;</span>
<span>} /* namespace thrust::unique&#95;eager&#95;future&#95;type&#95;detail */</span>
</code>

## Types

<h3 id="using-select">
Type Alias <code>thrust::unique&#95;eager&#95;future&#95;type&#95;detail::select</code>
</h3>

<code class="doxybook">
<span>template &lt;typename System,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>using <b>select</b> = decltype(unique&#95;eager&#95;future&#95;type&lt; T &gt;(std::declval&lt; System &gt;()));</span></code>

