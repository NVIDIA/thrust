---
title: thrust::mr::disjoint_unsynchronized_pool_resource::oversized_block_descriptor
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `thrust::mr::disjoint_unsynchronized_pool_resource::oversized_block_descriptor`

<code class="doxybook">
<span>struct thrust::mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::oversized&#95;block&#95;descriptor {</span>
<span>public:</span><span>&nbsp;&nbsp;std::size_t <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1disjoint__unsynchronized__pool__resource_1_1oversized__block__descriptor.html#variable-size">size</a></b>;</span>
<br>
<span>&nbsp;&nbsp;std::size_t <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1disjoint__unsynchronized__pool__resource_1_1oversized__block__descriptor.html#variable-alignment">alignment</a></b>;</span>
<br>
<span>&nbsp;&nbsp;void_ptr <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1disjoint__unsynchronized__pool__resource_1_1oversized__block__descriptor.html#variable-pointer">pointer</a></b>;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ bool </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1disjoint__unsynchronized__pool__resource_1_1oversized__block__descriptor.html#function-operator==">operator==</a></b>(const oversized_block_descriptor & other) const;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ bool </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1mr_1_1disjoint__unsynchronized__pool__resource_1_1oversized__block__descriptor.html#function-operator<">operator&lt;</a></b>(const oversized_block_descriptor & other) const;</span>
<span>};</span>
</code>

## Member Variables

<h3 id="variable-size">
Variable <code>thrust::mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::oversized&#95;block&#95;descriptor::size</code>
</h3>

<code class="doxybook">
<span>std::size_t <b>size</b>;</span></code>
<h3 id="variable-alignment">
Variable <code>thrust::mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::oversized&#95;block&#95;descriptor::alignment</code>
</h3>

<code class="doxybook">
<span>std::size_t <b>alignment</b>;</span></code>
<h3 id="variable-pointer">
Variable <code>thrust::mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::oversized&#95;block&#95;descriptor::pointer</code>
</h3>

<code class="doxybook">
<span>void_ptr <b>pointer</b>;</span></code>

## Member Functions

<h3 id="function-operator==">
Function <code>thrust::mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::oversized&#95;block&#95;descriptor::operator==</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ bool </span><span><b>operator==</b>(const oversized_block_descriptor & other) const;</span></code>
<h3 id="function-operator<">
Function <code>thrust::mr::disjoint&#95;unsynchronized&#95;pool&#95;resource::oversized&#95;block&#95;descriptor::operator&lt;</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ bool </span><span><b>operator<</b>(const oversized_block_descriptor & other) const;</span></code>

