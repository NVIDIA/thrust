---
title: thrust::proclaim_contiguous_iterator
summary: Customization point that can be customized to indicate that an iterator type Iterator satisfies ContiguousIterator, aka it points to elements that are contiguous in memory. 
parent: Type Traits
grand_parent: Utility
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `thrust::proclaim_contiguous_iterator`

Customization point that can be customized to indicate that an iterator type <code>Iterator</code> satisfies <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>, aka it points to elements that are contiguous in memory. 

**Inherits From**:
`false_type`

**See**:
* is_contiguous_iterator 
* <a href="{{ site.baseurl }}/api/groups/group__type__traits.html#define-thrust-proclaim-contiguous-iterator">THRUST_PROCLAIM_CONTIGUOUS_ITERATOR</a>

<code class="doxybook">
<span>#include <thrust/type_traits/is_contiguous_iterator.h></span><br>
<span>template &lt;typename Iterator&gt;</span>
<span>struct thrust::proclaim&#95;contiguous&#95;iterator {</span>
<span>};</span>
</code>

