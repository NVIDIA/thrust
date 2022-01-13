---
title: thrust::mr::allocator::rebind
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `thrust::mr::allocator::rebind`

The <code>rebind</code> metafunction provides the type of an <code>allocator</code> instantiated with another type.

**Template Parameters**:
**`U`**: the other type to use for instantiation. 

<code class="doxybook">
<span>#include <thrust/mr/allocator.h></span><br>
<span>template &lt;typename U&gt;</span>
<span>struct thrust::mr::allocator::rebind {</span>
<span>public:</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1allocator_1_1rebind.html#typedef-other">other</a></b>;</span>
<span>};</span>
</code>

## Member Types

<h3 id="typedef-other">
Typedef <code>thrust::mr::allocator::rebind::other</code>
</h3>

<code class="doxybook">
<span>typedef <a href="{{ site.baseurl }}/api/classes/classthrust_1_1mr_1_1allocator.html">allocator</a>< U, MR ><b>other</b>;</span></code>
The typedef <code>other</code> gives the type of the rebound <code>allocator</code>. 


