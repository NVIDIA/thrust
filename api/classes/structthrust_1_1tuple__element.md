---
title: thrust::tuple_element
parent: Tuple
grand_parent: Utility
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `thrust::tuple_element`

This convenience metafunction is included for compatibility with <code>tuple</code>. It returns either the type of a <code>pair's</code><code>first&#95;type</code> or <code>second&#95;type</code> in its nested type, <code>type</code>.


This metafunction returns the type of a <code>tuple's</code><code>N</code>th element.

**Template Parameters**:
* **`N`** This parameter selects the member of interest. 
* **`T`** A <code>pair</code> type of interest.
* **`N`** This parameter selects the element of interest. 
* **`T`** A <code>tuple</code> type of interest.

**See**:
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1pair.html">pair</a>
* <a href="{{ site.baseurl }}/api/classes/classthrust_1_1tuple.html">tuple</a>

<code class="doxybook">
<span>template &lt;size_t N,</span>
<span>&nbsp;&nbsp;class T&gt;</span>
<span>struct thrust::tuple&#95;element {</span>
<span>};</span>
</code>

