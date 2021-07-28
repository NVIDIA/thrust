---
title: device_allocator::rebind
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `device_allocator::rebind`

The <code>rebind</code> metafunction provides the type of a <code><a href="/api/classes/classdevice__allocator.html">device&#95;allocator</a></code> instantiated with another type.

**Template Parameters**:
**`U`**: the other type to use for instantiation. 

<code class="doxybook">
<span>#include <thrust/device_allocator.h></span><br>
<span>template &lt;typename U&gt;</span>
<span>struct device&#95;allocator::rebind {</span>
<span>public:</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="/api/classes/structdevice__allocator_1_1rebind.html#typedef-other">other</a></b>;</span>
<span>};</span>
</code>

## Member Types

<h3 id="typedef-other">
Typedef <code>device&#95;allocator::rebind::other</code>
</h3>

<code class="doxybook">
<span>typedef <a href="/api/classes/classdevice__allocator.html">device_allocator</a>< U ><b>other</b>;</span></code>
The typedef <code>other</code> gives the type of the rebound <code><a href="/api/classes/classdevice__allocator.html">device&#95;allocator</a></code>. 


