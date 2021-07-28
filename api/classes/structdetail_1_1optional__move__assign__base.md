---
title: detail::optional_move_assign_base
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `detail::optional_move_assign_base`

**Inherits From**:
`detail::optional_copy_assign_base< T >`

<code class="doxybook">
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;bool = std::is&#95;trivially&#95;destructible&lt; T &gt;::value && std::is&#95;trivially&#95;move&#95;constructible&lt; T &gt;::value && std::is&#95;trivially&#95;move&#95;assignable&lt; T &gt;::value&gt;</span>
<span>struct detail::optional&#95;move&#95;assign&#95;base {</span>
<span>};</span>
</code>

