---
title: thrust::device_new_allocator::rebind
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `thrust::device_new_allocator::rebind`

The <code>rebind</code> metafunction provides the type of a <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__new__allocator.html">device&#95;new&#95;allocator</a></code> instantiated with another type.

**Template Parameters**:
**`U`**: The other type to use for instantiation. 

<code class="doxybook">
<span>#include <thrust/device_new_allocator.h></span><br>
<span>template &lt;typename U&gt;</span>
<span>struct thrust::device&#95;new&#95;allocator::rebind {</span>
<span>public:</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1device__new__allocator_1_1rebind.html#typedef-other">other</a></b>;</span>
<span>};</span>
</code>

## Member Types

<h3 id="typedef-other">
Typedef <code>thrust::device&#95;new&#95;allocator::rebind::other</code>
</h3>

<code class="doxybook">
<span>typedef <a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__new__allocator.html">device_new_allocator</a>< U ><b>other</b>;</span></code>
The typedef <code>other</code> gives the type of the rebound <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__new__allocator.html">device&#95;new&#95;allocator</a></code>. 


