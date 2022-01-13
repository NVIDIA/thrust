---
title: thrust::remove_cvref
summary: UnaryTypeTrait that removes const-volatile qualifiers and references from T. Equivalent to remove_cv_t<remove_reference_t<T>>. 
parent: Type Traits
grand_parent: Utility
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `thrust::remove_cvref`

<a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that removes <a href="https://en.cppreference.com/w/cpp/language/cv">const-volatile qualifiers</a> and <a href="https://en.cppreference.com/w/cpp/language/reference">references</a> from <code>T</code>. Equivalent to <code>remove&#95;cv&#95;t&lt;remove&#95;reference&#95;t&lt;T&gt;&gt;</code>. 

**See**:
* <a href="https://en.cppreference.com/w/cpp/types/remove_cvref">std::remove_cvref</a>
* <a href="https://en.cppreference.com/w/cpp/types/remove_cv">std::remove_cv</a>
* <a href="https://en.cppreference.com/w/cpp/types/remove_cv">std::remove_const</a>
* <a href="https://en.cppreference.com/w/cpp/types/remove_cv">std::remove_volatile</a>
* <a href="https://en.cppreference.com/w/cpp/types/remove_cv">std::remove_reference</a>

<code class="doxybook">
<span>#include <thrust/type_traits/remove_cvref.h></span><br>
<span>template &lt;typename T&gt;</span>
<span>struct thrust::remove&#95;cvref {</span>
<span>public:</span><span>&nbsp;&nbsp;using <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1remove__cvref.html#using-type">type</a></b> = <i>see below</i>;</span>
<span>};</span>
</code>

## Member Types

<h3 id="using-type">
Type Alias <code>thrust::remove&#95;cvref::type</code>
</h3>

<code class="doxybook">
<span>using <b>type</b> = typename std::remove&#95;cv&lt; typename std::remove&#95;reference&lt; T &gt;::type &gt;::type;</span></code>

