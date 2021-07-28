---
title: detail::optional_copy_assign_base
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `detail::optional_copy_assign_base`

**Inherits From**:
`detail::optional_move_base< T >`

**Inherited By**:
* [`detail::optional_move_assign_base< T, bool >`](/api/classes/structdetail_1_1optional__move__assign__base.html)
* [`detail::optional_move_assign_base< T, false >`](/api/classes/structdetail_1_1optional__move__assign__base_3_01t_00_01false_01_4.html)

<code class="doxybook">
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;bool = std::is&#95;trivially&#95;copy&#95;assignable&lt; T &gt;::value && std::is&#95;trivially&#95;copy&#95;constructible&lt; T &gt;::value && std::is&#95;trivially&#95;destructible&lt; T &gt;::value&gt;</span>
<span>struct detail::optional&#95;copy&#95;assign&#95;base {</span>
<span>};</span>
</code>

