---
title: cuda
nav_exclude: true
has_children: true
has_toc: false
---

# Namespace `cuda`

<code class="doxybook">
<span>namespace cuda {</span>
<br>
<span>using <b><a href="/thrust/api/namespaces/namespacecuda.html#using-event">event</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;typename T&gt;</span>
<span>using <b><a href="/thrust/api/namespaces/namespacecuda.html#using-future">future</a></b> = <i>see below</i>;</span>
<span>} /* namespace cuda */</span>
</code>

## Types

<h3 id="using-event">
Type Alias <code>event</code>
</h3>

<code class="doxybook">
<span>using <b>event</b> = unique&#95;eager&#95;event;</span></code>
<h3 id="using-future">
Type Alias <code>future</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>future</b> = unique&#95;eager&#95;future&lt; T &gt;;</span></code>

