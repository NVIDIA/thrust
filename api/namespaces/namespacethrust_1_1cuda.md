---
title: thrust::cuda
summary: thrust::cuda is a top-level alias for thrust::system::cuda. 
nav_exclude: true
has_children: true
has_toc: false
---

# Namespace `thrust::cuda`

<code class="doxybook">
<span>namespace thrust::cuda {</span>
<br>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1cuda.html#using-event">event</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1cuda.html#using-future">future</a></b> = <i>see below</i>;</span>
<span>} /* namespace thrust::cuda */</span>
</code>

## Types

<h3 id="using-event">
Type Alias <code>thrust::cuda::event</code>
</h3>

<code class="doxybook">
<span>using <b>event</b> = unique&#95;eager&#95;event;</span></code>
<h3 id="using-future">
Type Alias <code>thrust::cuda::future</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>future</b> = unique&#95;eager&#95;future&lt; T &gt;;</span></code>

