---
title: proclaim_trivially_relocatable
summary: customization point that can be specialized customized to indicate that a type T is TriviallyRelocatable, aka it can be bitwise copied with a facility like std::memcpy. 
parent: Type Traits
grand_parent: Utility
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `proclaim_trivially_relocatable`

<a href="http://eel.is/c++draft/namespace.std#def:customization_point">_customization point_</a> that can be specialized customized to indicate that a type <code>T</code> is <a href="https://wg21.link/P1144">_TriviallyRelocatable_</a>, aka it can be bitwise copied with a facility like <a href="https://en.cppreference.com/w/cpp/string/byte/memcpy"><code>std::memcpy</code></a>. 

**Inherits From**:
`false_type`

**See**:
* <a href="/thrust/api/groups/group__type__traits.html#using-is_indirectly_trivially_relocatable_to">is_indirectly_trivially_relocatable_to</a>
* <a href="/thrust/api/groups/group__type__traits.html#using-is_trivially_relocatable">is_trivially_relocatable</a>
* <a href="/thrust/api/groups/group__type__traits.html#using-is_trivially_relocatable_to">is_trivially_relocatable_to</a>
* <a href="/thrust/api/groups/group__type__traits.html#define-thrust_proclaim_trivially_relocatable">THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE</a>

<code class="doxybook">
<span>#include <thrust/type_traits/is_trivially_relocatable.h></span><br>
<span>template &lt;typename T&gt;</span>
<span>struct proclaim&#95;trivially&#95;relocatable {</span>
<span>};</span>
</code>

