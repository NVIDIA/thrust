---
title: unique_eager_future_type_detail
nav_exclude: true
has_children: true
has_toc: false
---

# Namespace `unique_eager_future_type_detail`

<code class="doxybook">
<span>namespace unique&#95;eager&#95;future&#95;type&#95;detail {</span>
<br>
<span>template &lt;typename System,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>using <b><a href="/api/namespaces/namespaceunique__eager__future__type__detail.html#using-select">select</a></b> = <i>see below</i>;</span>
<span>} /* namespace unique&#95;eager&#95;future&#95;type&#95;detail */</span>
</code>

## Types

<h3 id="using-select">
Type Alias <code>select</code>
</h3>

<code class="doxybook">
<span>template &lt;typename System,</span>
<span>&nbsp;&nbsp;typename T&gt;</span>
<span>using <b>select</b> = decltype(unique&#95;eager&#95;future&#95;type&lt; T &gt;(std::declval&lt; System &gt;()));</span></code>

