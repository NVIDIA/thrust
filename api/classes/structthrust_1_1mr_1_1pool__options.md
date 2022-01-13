---
title: thrust::mr::pool_options
parent: Memory Resources
grand_parent: Memory Management
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `thrust::mr::pool_options`

A type used for configuring pooling resource adaptors, to fine-tune their behavior and parameters. 

<code class="doxybook">
<span>#include <thrust/mr/pool_options.h></span><br>
<span>struct thrust::mr::pool&#95;options {</span>
<span>public:</span><span>&nbsp;&nbsp;std::size_t <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1pool__options.html#variable-min-blocks-per-chunk">min&#95;blocks&#95;per&#95;chunk</a></b>;</span>
<br>
<span>&nbsp;&nbsp;std::size_t <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1pool__options.html#variable-min-bytes-per-chunk">min&#95;bytes&#95;per&#95;chunk</a></b>;</span>
<br>
<span>&nbsp;&nbsp;std::size_t <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1pool__options.html#variable-max-blocks-per-chunk">max&#95;blocks&#95;per&#95;chunk</a></b>;</span>
<br>
<span>&nbsp;&nbsp;std::size_t <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1pool__options.html#variable-max-bytes-per-chunk">max&#95;bytes&#95;per&#95;chunk</a></b>;</span>
<br>
<span>&nbsp;&nbsp;std::size_t <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1pool__options.html#variable-smallest-block-size">smallest&#95;block&#95;size</a></b>;</span>
<br>
<span>&nbsp;&nbsp;std::size_t <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1pool__options.html#variable-largest-block-size">largest&#95;block&#95;size</a></b>;</span>
<br>
<span>&nbsp;&nbsp;std::size_t <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1pool__options.html#variable-alignment">alignment</a></b>;</span>
<br>
<span>&nbsp;&nbsp;bool <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1pool__options.html#variable-cache-oversized">cache&#95;oversized</a></b>;</span>
<br>
<span>&nbsp;&nbsp;std::size_t <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1pool__options.html#variable-cached-size-cutoff-factor">cached&#95;size&#95;cutoff&#95;factor</a></b>;</span>
<br>
<span>&nbsp;&nbsp;std::size_t <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1pool__options.html#variable-cached-alignment-cutoff-factor">cached&#95;alignment&#95;cutoff&#95;factor</a></b>;</span>
<br>
<span>&nbsp;&nbsp;bool </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1pool__options.html#function-validate">validate</a></b>() const;</span>
<span>};</span>
</code>

## Member Variables

<h3 id="variable-min-blocks-per-chunk">
Variable <code>thrust::mr::pool&#95;options::min&#95;blocks&#95;per&#95;chunk</code>
</h3>

<code class="doxybook">
<span>std::size_t <b>min_blocks_per_chunk</b>;</span></code>
The minimal number of blocks, i.e. pieces of memory handed off to the user from a pool of a given size, in a single chunk allocated from upstream. 

<h3 id="variable-min-bytes-per-chunk">
Variable <code>thrust::mr::pool&#95;options::min&#95;bytes&#95;per&#95;chunk</code>
</h3>

<code class="doxybook">
<span>std::size_t <b>min_bytes_per_chunk</b>;</span></code>
The minimal number of bytes in a single chunk allocated from upstream. 

<h3 id="variable-max-blocks-per-chunk">
Variable <code>thrust::mr::pool&#95;options::max&#95;blocks&#95;per&#95;chunk</code>
</h3>

<code class="doxybook">
<span>std::size_t <b>max_blocks_per_chunk</b>;</span></code>
The maximal number of blocks, i.e. pieces of memory handed off to the user from a pool of a given size, in a single chunk allocated from upstream. 

<h3 id="variable-max-bytes-per-chunk">
Variable <code>thrust::mr::pool&#95;options::max&#95;bytes&#95;per&#95;chunk</code>
</h3>

<code class="doxybook">
<span>std::size_t <b>max_bytes_per_chunk</b>;</span></code>
The maximal number of bytes in a single chunk allocated from upstream. 

<h3 id="variable-smallest-block-size">
Variable <code>thrust::mr::pool&#95;options::smallest&#95;block&#95;size</code>
</h3>

<code class="doxybook">
<span>std::size_t <b>smallest_block_size</b>;</span></code>
The size of blocks in the smallest pool covered by the pool resource. All allocation requests below this size will be rounded up to this size. 

<h3 id="variable-largest-block-size">
Variable <code>thrust::mr::pool&#95;options::largest&#95;block&#95;size</code>
</h3>

<code class="doxybook">
<span>std::size_t <b>largest_block_size</b>;</span></code>
The size of blocks in the largest pool covered by the pool resource. All allocation requests above this size will be considered oversized, allocated directly from upstream (and not from a pool), and cached only of <code>cache&#95;oversized</code> is true. 

<h3 id="variable-alignment">
Variable <code>thrust::mr::pool&#95;options::alignment</code>
</h3>

<code class="doxybook">
<span>std::size_t <b>alignment</b>;</span></code>
The alignment of all blocks in internal pools of the pool resource. All allocation requests above this alignment will be considered oversized, allocated directly from upstream (and not from a pool), and cached only of <code>cache&#95;oversized</code> is true. 

<h3 id="variable-cache-oversized">
Variable <code>thrust::mr::pool&#95;options::cache&#95;oversized</code>
</h3>

<code class="doxybook">
<span>bool <b>cache_oversized</b>;</span></code>
Decides whether oversized and overaligned blocks are cached for later use, or immediately return it to the upstream resource. 

<h3 id="variable-cached-size-cutoff-factor">
Variable <code>thrust::mr::pool&#95;options::cached&#95;size&#95;cutoff&#95;factor</code>
</h3>

<code class="doxybook">
<span>std::size_t <b>cached_size_cutoff_factor</b>;</span></code>
The size factor at which a cached allocation is considered too ridiculously oversized to use to fulfill an allocation request. For instance: the user requests an allocation of size 1024 bytes. A block of size 32 * 1024 bytes is cached. If <code>cached&#95;size&#95;cutoff&#95;factor</code> is 32 or less, this block will be considered too big for that allocation request. 

<h3 id="variable-cached-alignment-cutoff-factor">
Variable <code>thrust::mr::pool&#95;options::cached&#95;alignment&#95;cutoff&#95;factor</code>
</h3>

<code class="doxybook">
<span>std::size_t <b>cached_alignment_cutoff_factor</b>;</span></code>
The alignment factor at which a cached allocation is considered too ridiculously overaligned to use to fulfill an allocation request. For instance: the user requests an allocation aligned to 32 bytes. A block aligned to 1024 bytes is cached. If <code>cached&#95;size&#95;cutoff&#95;factor</code> is 32 or less, this block will be considered too overaligned for that allocation request. 


## Member Functions

<h3 id="function-validate">
Function <code>thrust::mr::pool&#95;options::validate</code>
</h3>

<code class="doxybook">
<span>bool </span><span><b>validate</b>() const;</span></code>
Checks if the options are self-consistent.

/returns true if the options are self-consitent, false otherwise. 


