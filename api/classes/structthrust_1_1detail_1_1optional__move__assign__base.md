---
title: thrust::detail::optional_move_assign_base
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `thrust::detail::optional_move_assign_base`

**Inherits From**:
`thrust::detail::optional_copy_assign_base< T >`

**Inherited By**:
[`thrust::optional< T >`]({{ site.baseurl }}/api/classes/classthrust_1_1optional.html)

<code class="doxybook">
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;bool = std::is&#95;trivially&#95;destructible&lt; T &gt;::value && std::is&#95;trivially&#95;move&#95;constructible&lt; T &gt;::value && std::is&#95;trivially&#95;move&#95;assignable&lt; T &gt;::value&gt;</span>
<span>struct thrust::detail::optional&#95;move&#95;assign&#95;base {</span>
<span>};</span>
</code>

