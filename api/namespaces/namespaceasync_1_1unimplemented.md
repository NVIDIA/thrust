---
title: async::unimplemented
nav_exclude: true
has_children: true
has_toc: false
---

# Namespace `async::unimplemented`

<code class="doxybook">
<span>namespace async::unimplemented {</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIt,</span>
<span>&nbsp;&nbsp;typename Sentinel,</span>
<span>&nbsp;&nbsp;typename OutputIt,</span>
<span>&nbsp;&nbsp;typename BinaryOp&gt;</span>
<span>event< DerivedPolicy > </span><span><b><a href="/api/namespaces/namespaceasync_1_1unimplemented.html#function-async_inclusive_scan">async&#95;inclusive&#95;scan</a></b>(thrust::execution_policy< DerivedPolicy > &,</span>
<span>&nbsp;&nbsp;ForwardIt,</span>
<span>&nbsp;&nbsp;Sentinel,</span>
<span>&nbsp;&nbsp;OutputIt,</span>
<span>&nbsp;&nbsp;BinaryOp);</span>
<br>
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIt,</span>
<span>&nbsp;&nbsp;typename Sentinel,</span>
<span>&nbsp;&nbsp;typename OutputIt,</span>
<span>&nbsp;&nbsp;typename InitialValueType,</span>
<span>&nbsp;&nbsp;typename BinaryOp&gt;</span>
<span>event< DerivedPolicy > </span><span><b><a href="/api/namespaces/namespaceasync_1_1unimplemented.html#function-async_exclusive_scan">async&#95;exclusive&#95;scan</a></b>(thrust::execution_policy< DerivedPolicy > &,</span>
<span>&nbsp;&nbsp;ForwardIt,</span>
<span>&nbsp;&nbsp;Sentinel,</span>
<span>&nbsp;&nbsp;OutputIt,</span>
<span>&nbsp;&nbsp;InitialValueType,</span>
<span>&nbsp;&nbsp;BinaryOp);</span>
<span>} /* namespace async::unimplemented */</span>
</code>

## Functions

<h3 id="function-async_inclusive_scan">
Function <code>async::unimplemented::async&#95;inclusive&#95;scan</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIt,</span>
<span>&nbsp;&nbsp;typename Sentinel,</span>
<span>&nbsp;&nbsp;typename OutputIt,</span>
<span>&nbsp;&nbsp;typename BinaryOp&gt;</span>
<span>event< DerivedPolicy > </span><span><b>async_inclusive_scan</b>(thrust::execution_policy< DerivedPolicy > &,</span>
<span>&nbsp;&nbsp;ForwardIt,</span>
<span>&nbsp;&nbsp;Sentinel,</span>
<span>&nbsp;&nbsp;OutputIt,</span>
<span>&nbsp;&nbsp;BinaryOp);</span></code>
<h3 id="function-async_exclusive_scan">
Function <code>async::unimplemented::async&#95;exclusive&#95;scan</code>
</h3>

<code class="doxybook">
<span>template &lt;typename DerivedPolicy,</span>
<span>&nbsp;&nbsp;typename ForwardIt,</span>
<span>&nbsp;&nbsp;typename Sentinel,</span>
<span>&nbsp;&nbsp;typename OutputIt,</span>
<span>&nbsp;&nbsp;typename InitialValueType,</span>
<span>&nbsp;&nbsp;typename BinaryOp&gt;</span>
<span>event< DerivedPolicy > </span><span><b>async_exclusive_scan</b>(thrust::execution_policy< DerivedPolicy > &,</span>
<span>&nbsp;&nbsp;ForwardIt,</span>
<span>&nbsp;&nbsp;Sentinel,</span>
<span>&nbsp;&nbsp;OutputIt,</span>
<span>&nbsp;&nbsp;InitialValueType,</span>
<span>&nbsp;&nbsp;BinaryOp);</span></code>

