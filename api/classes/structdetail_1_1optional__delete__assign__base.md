---
title: detail::optional_delete_assign_base
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `detail::optional_delete_assign_base`

<code class="doxybook">
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;bool EnableCopy = (std::is&#95;copy&#95;constructible&lt;T&gt;::value &&                             std::is&#95;copy&#95;assignable&lt;T&gt;::value),</span>
<span>&nbsp;&nbsp;bool EnableMove = (std::is&#95;move&#95;constructible&lt;T&gt;::value &&                             std::is&#95;move&#95;assignable&lt;T&gt;::value)&gt;</span>
<span>struct detail::optional&#95;delete&#95;assign&#95;base {</span>
<span>public:</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ </span><span>&nbsp;&nbsp;<b><a href="/api/classes/structdetail_1_1optional__delete__assign__base.html#function-optional_delete_assign_base">optional&#95;delete&#95;assign&#95;base</a></b>() = default;</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ </span><span>&nbsp;&nbsp;<b><a href="/api/classes/structdetail_1_1optional__delete__assign__base.html#function-optional_delete_assign_base">optional&#95;delete&#95;assign&#95;base</a></b>(const optional_delete_assign_base &) = default;</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ </span><span>&nbsp;&nbsp;<b><a href="/api/classes/structdetail_1_1optional__delete__assign__base.html#function-optional_delete_assign_base">optional&#95;delete&#95;assign&#95;base</a></b>(optional_delete_assign_base &&) = default;</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ optional_delete_assign_base & </span><span>&nbsp;&nbsp;<b><a href="/api/classes/structdetail_1_1optional__delete__assign__base.html#function-operator=">operator=</a></b>(const optional_delete_assign_base &) = default;</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ optional_delete_assign_base & </span><span>&nbsp;&nbsp;<b><a href="/api/classes/structdetail_1_1optional__delete__assign__base.html#function-operator=">operator=</a></b>(optional_delete_assign_base &&) = default;</span>
<span>};</span>
</code>

## Member Functions

<h3 id="function-optional_delete_assign_base">
Function <code>detail::optional&#95;delete&#95;assign&#95;base::&gt;::optional&#95;delete&#95;assign&#95;base</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ </span><span><b>optional_delete_assign_base</b>() = default;</span></code>
<h3 id="function-optional_delete_assign_base">
Function <code>detail::optional&#95;delete&#95;assign&#95;base::&gt;::optional&#95;delete&#95;assign&#95;base</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ </span><span><b>optional_delete_assign_base</b>(const optional_delete_assign_base &) = default;</span></code>
<h3 id="function-optional_delete_assign_base">
Function <code>detail::optional&#95;delete&#95;assign&#95;base::&gt;::optional&#95;delete&#95;assign&#95;base</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ </span><span><b>optional_delete_assign_base</b>(optional_delete_assign_base &&) = default;</span></code>
<h3 id="function-operator=">
Function <code>detail::optional&#95;delete&#95;assign&#95;base::&gt;::operator=</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ optional_delete_assign_base & </span><span><b>operator=</b>(const optional_delete_assign_base &) = default;</span></code>
<h3 id="function-operator=">
Function <code>detail::optional&#95;delete&#95;assign&#95;base::&gt;::operator=</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ optional_delete_assign_base & </span><span><b>operator=</b>(optional_delete_assign_base &&) = default;</span></code>

