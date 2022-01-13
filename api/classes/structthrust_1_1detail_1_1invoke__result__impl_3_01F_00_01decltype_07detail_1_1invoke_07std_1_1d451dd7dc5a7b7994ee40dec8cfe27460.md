---
title: thrust::detail::invoke_result_impl< F, decltype(detail::invoke(std::declval< F >(), std::declval< Us >()...), void()), Us... >
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `thrust::detail::invoke_result_impl< F, decltype(detail::invoke(std::declval< F >(), std::declval< Us >()...), void()), Us... >`

<code class="doxybook">
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class... Us&gt;</span>
<span>struct thrust::detail::invoke&#95;result&#95;impl&lt; F, decltype(detail::invoke(std::declval&lt; F &gt;(), std::declval&lt; Us &gt;()...), void()), Us... &gt; {</span>
<span>public:</span><span>&nbsp;&nbsp;using <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1invoke__result__impl_3_01f_00_01decltype_07detail_1_1invoke_07std_1.html#using-type">type</a></b> = <i>see below</i>;</span>
<span>};</span>
</code>

## Member Types

<h3 id="using-type">
Type Alias <code>thrust::detail::invoke&#95;result&#95;impl&lt; F, decltype(detail::invoke(std::declval&lt; F &gt;(), std::declval&lt; Us &gt;()...), void()), Us... &gt;::type</code>
</h3>

<code class="doxybook">
<span>using <b>type</b> = decltype(detail::invoke(std::declval&lt; F &gt;(), std::declval&lt; Us &gt;()...));</span></code>

