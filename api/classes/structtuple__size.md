---
title: tuple_size
parent: Tuple
grand_parent: Utility
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `tuple_size`

This convenience metafunction is included for compatibility with <code>tuple</code>. It returns <code>2</code>, the number of elements of a <code>pair</code>, in its nested data member, <code>value</code>.


This metafunction returns the number of elements of a <code>tuple</code> type of interest.

**Template Parameters**:
* **`Pair`** A <code>pair</code> type of interest.
* **`T`** A <code>tuple</code> type of interest.

**See**:
* <a href="/api/classes/structpair.html">pair</a>
* <a href="/api/classes/classtuple.html">tuple</a>

<code class="doxybook">
<span>#include <thrust/pair.h></span><br>
<span>template &lt;typename Pair&gt;</span>
<span>struct tuple&#95;size {</span>
<span>};</span>
</code>

