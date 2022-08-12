---
title: thrust::detail
summary: \exclude 
parent: Function Object Adaptors
grand_parent: Function Objects
nav_exclude: true
has_children: true
has_toc: false
---

# Namespace `thrust::detail`

<code class="doxybook">
<span>namespace thrust::detail {</span>
<br>
<span>namespace <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail_1_1permutation__iterator__base.html">thrust::detail::permutation&#95;iterator&#95;base</a></b> { <i>…</i> }</span>
<br>
<span>namespace <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail_1_1zip__detail.html">thrust::detail::zip&#95;detail</a></b> { <i>…</i> }</span>
<br>
<span>template &lt;class...&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1conjunction.html">conjunction</a></b>;</span>
<br>
<span>template &lt;class B&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1conjunction_3_01b_01_4.html">conjunction&lt; B &gt;</a></b>;</span>
<br>
<span>template &lt;class B,</span>
<span>&nbsp;&nbsp;class... Bs&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1conjunction_3_01b_00_01bs_8_8_8_01_4.html">conjunction&lt; B, Bs... &gt;</a></b>;</span>
<br>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1i__am__secret.html">i&#95;am&#95;secret</a></b>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class U = T&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1is__nothrow__swappable.html">is&#95;nothrow&#95;swappable</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1is__optional__impl.html">is&#95;optional&#95;impl</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1is__optional__impl_3_01optional_3_01t_01_4_01_4.html">is&#95;optional&#95;impl&lt; optional&lt; T &gt; &gt;</a></b>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class U = T&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1is__swappable.html">is&#95;swappable</a></b>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;bool = std::is&#95;trivially&#95;copy&#95;assignable&lt; T &gt;::value && std::is&#95;trivially&#95;copy&#95;constructible&lt; T &gt;::value && std::is&#95;trivially&#95;destructible&lt; T &gt;::value&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__copy__assign__base.html">optional&#95;copy&#95;assign&#95;base</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__copy__assign__base_3_01t_00_01false_01_4.html">optional&#95;copy&#95;assign&#95;base&lt; T, false &gt;</a></b>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;bool = std::is&#95;trivially&#95;copy&#95;constructible&lt; T &gt;::value&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__copy__base.html">optional&#95;copy&#95;base</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__copy__base_3_01t_00_01false_01_4.html">optional&#95;copy&#95;base&lt; T, false &gt;</a></b>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;bool EnableCopy = (std::is&#95;copy&#95;constructible&lt;T&gt;::value &&                             std::is&#95;copy&#95;assignable&lt;T&gt;::value),</span>
<span>&nbsp;&nbsp;bool EnableMove = (std::is&#95;move&#95;constructible&lt;T&gt;::value &&                             std::is&#95;move&#95;assignable&lt;T&gt;::value)&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__delete__assign__base.html">optional&#95;delete&#95;assign&#95;base</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__delete__assign__base_3_01t_00_01false_00_01false_01_4.html">optional&#95;delete&#95;assign&#95;base&lt; T, false, false &gt;</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__delete__assign__base_3_01t_00_01false_00_01true_01_4.html">optional&#95;delete&#95;assign&#95;base&lt; T, false, true &gt;</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__delete__assign__base_3_01t_00_01true_00_01false_01_4.html">optional&#95;delete&#95;assign&#95;base&lt; T, true, false &gt;</a></b>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;bool EnableCopy = std::is&#95;copy&#95;constructible&lt;T&gt;::value,</span>
<span>&nbsp;&nbsp;bool EnableMove = std::is&#95;move&#95;constructible&lt;T&gt;::value&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__delete__ctor__base.html">optional&#95;delete&#95;ctor&#95;base</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__delete__ctor__base_3_01t_00_01false_00_01false_01_4.html">optional&#95;delete&#95;ctor&#95;base&lt; T, false, false &gt;</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__delete__ctor__base_3_01t_00_01false_00_01true_01_4.html">optional&#95;delete&#95;ctor&#95;base&lt; T, false, true &gt;</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__delete__ctor__base_3_01t_00_01true_00_01false_01_4.html">optional&#95;delete&#95;ctor&#95;base&lt; T, true, false &gt;</a></b>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;bool = std::is&#95;trivially&#95;destructible&lt; T &gt;::value && std::is&#95;trivially&#95;move&#95;constructible&lt; T &gt;::value && std::is&#95;trivially&#95;move&#95;assignable&lt; T &gt;::value&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__move__assign__base.html">optional&#95;move&#95;assign&#95;base</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__move__assign__base_3_01t_00_01false_01_4.html">optional&#95;move&#95;assign&#95;base&lt; T, false &gt;</a></b>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;bool = std::is&#95;trivially&#95;move&#95;constructible&lt; T &gt;::value&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__move__base.html">optional&#95;move&#95;base</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__move__base_3_01t_00_01false_01_4.html">optional&#95;move&#95;base&lt; T, false &gt;</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__operations__base.html">optional&#95;operations&#95;base</a></b>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;bool = ::std::is&#95;trivially&#95;destructible&lt;T&gt;::value&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__storage__base.html">optional&#95;storage&#95;base</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__storage__base_3_01t_00_01true_01_4.html">optional&#95;storage&#95;base&lt; T, true &gt;</a></b>;</span>
<br>
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class = void,</span>
<span>&nbsp;&nbsp;class... U&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1returns__void__impl.html">returns&#95;void&#95;impl</a></b>;</span>
<br>
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class... U&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1returns__void__impl_3_01f_00_01void__t_3_01invoke__result__t_3_01f_00_92d99556b5b7d67309a0911581ace58d.html">returns&#95;void&#95;impl&lt; F, void&#95;t&lt; invoke&#95;result&#95;t&lt; F, U... &gt; &gt;, U... &gt;</a></b>;</span>
<br>
<span>template &lt;class...&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1voider.html">voider</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail.html#using-remove-const-t">remove&#95;const&#95;t</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail.html#using-remove-reference-t">remove&#95;reference&#95;t</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail.html#using-decay-t">decay&#95;t</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;bool E,</span>
<span>&nbsp;&nbsp;class T = void&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail.html#using-enable-if-t">enable&#95;if&#95;t</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;bool B,</span>
<span>&nbsp;&nbsp;class T,</span>
<span>&nbsp;&nbsp;class F&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail.html#using-conditional-t">conditional&#95;t</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class... Ts&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail.html#using-void-t">void&#95;t</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail.html#using-is-optional">is&#95;optional</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class U&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail.html#using-fixup-void">fixup&#95;void</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class U,</span>
<span>&nbsp;&nbsp;class = invoke&#95;result&#95;t&lt;F, U&gt;&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail.html#using-get-map-return">get&#95;map&#95;return</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class... U&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail.html#using-returns-void">returns&#95;void</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class... U&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail.html#using-enable-if-ret-void">enable&#95;if&#95;ret&#95;void</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class... U&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail.html#using-disable-if-ret-void">disable&#95;if&#95;ret&#95;void</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class U&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail.html#using-enable-forward-value">enable&#95;forward&#95;value</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class U,</span>
<span>&nbsp;&nbsp;class Other&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail.html#using-enable-from-other">enable&#95;from&#95;other</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class U&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail.html#using-enable-assign-forward">enable&#95;assign&#95;forward</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class U,</span>
<span>&nbsp;&nbsp;class Other&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail.html#using-enable-assign-from-other">enable&#95;assign&#95;from&#95;other</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;typename Allocator,</span>
<span>&nbsp;&nbsp;typename Pointer&gt;</span>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail.html#function-allocator-delete-impl">allocator&#95;delete&#95;impl</a></b>(Allocator const & alloc,</span>
<span>&nbsp;&nbsp;Pointer p,</span>
<span>&nbsp;&nbsp;std::false_type);</span>
<br>
<span>template &lt;typename Allocator,</span>
<span>&nbsp;&nbsp;typename Pointer&gt;</span>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail.html#function-allocator-delete-impl">allocator&#95;delete&#95;impl</a></b>(Allocator const & alloc,</span>
<span>&nbsp;&nbsp;Pointer p,</span>
<span>&nbsp;&nbsp;std::true_type);</span>
<br>
<span>template &lt;typename Allocator,</span>
<span>&nbsp;&nbsp;typename Pointer,</span>
<span>&nbsp;&nbsp;typename Size&gt;</span>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail.html#function-array-allocator-delete-impl">array&#95;allocator&#95;delete&#95;impl</a></b>(Allocator const & alloc,</span>
<span>&nbsp;&nbsp;Pointer p,</span>
<span>&nbsp;&nbsp;Size count,</span>
<span>&nbsp;&nbsp;std::false_type);</span>
<br>
<span>template &lt;typename Allocator,</span>
<span>&nbsp;&nbsp;typename Pointer,</span>
<span>&nbsp;&nbsp;typename Size&gt;</span>
<span>void </span><span><b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail.html#function-array-allocator-delete-impl">array&#95;allocator&#95;delete&#95;impl</a></b>(Allocator const & alloc,</span>
<span>&nbsp;&nbsp;Pointer p,</span>
<span>&nbsp;&nbsp;Size count,</span>
<span>&nbsp;&nbsp;std::true_type);</span>
<br>
<span>template &lt;typename Fn,</span>
<span>&nbsp;&nbsp;typename... Args,</span>
<span>&nbsp;&nbsp;typename = enable&#95;if&#95;t&lt;std::is&#95;member&#95;pointer&lt;decay&#95;t&lt;Fn&gt;&gt;::value&gt;,</span>
<span>&nbsp;&nbsp;int = 0&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ constexpr auto </span><span><b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail.html#function-invoke">invoke</a></b>(Fn && f,</span>
<span>&nbsp;&nbsp;Args &&... args);</span>
<br>
<span>template &lt;typename Fn,</span>
<span>&nbsp;&nbsp;typename... Args,</span>
<span>&nbsp;&nbsp;typename = enable&#95;if&#95;t&lt;!std::is&#95;member&#95;pointer&lt;decay&#95;t&lt;Fn&gt;&gt;::value&gt;&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ constexpr auto </span><span><b><a href="{{ site.baseurl }}/api/namespaces/namespacethrust_1_1detail.html#function-invoke">invoke</a></b>(Fn && f,</span>
<span>&nbsp;&nbsp;Args &&... args);</span>
<span>} /* namespace thrust::detail */</span>
</code>

## Member Classes

<h3 id="struct-thrustdetailconjunction">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1conjunction.html">Struct <code>thrust::detail::conjunction</code>
</a>
</h3>

**Inherits From**:
`std::true_type`

<h3 id="struct-thrustdetailconjunction<-b->">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1conjunction_3_01b_01_4.html">Struct <code>thrust::detail::conjunction&lt; B &gt;</code>
</a>
</h3>

**Inherits From**:
`B`

<h3 id="struct-thrustdetailconjunction<-b,-bs...->">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1conjunction_3_01b_00_01bs_8_8_8_01_4.html">Struct <code>thrust::detail::conjunction&lt; B, Bs... &gt;</code>
</a>
</h3>

**Inherits From**:
`std::conditional::type`

<h3 id="struct-thrustdetaili-am-secret">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1i__am__secret.html">Struct <code>thrust::detail::i&#95;am&#95;secret</code>
</a>
</h3>

<h3 id="struct-thrustdetailis-nothrow-swappable">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1is__nothrow__swappable.html">Struct <code>thrust::detail::is&#95;nothrow&#95;swappable</code>
</a>
</h3>

**Inherits From**:
`std::true_type`

<h3 id="struct-thrustdetailis-optional-impl">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1is__optional__impl.html">Struct <code>thrust::detail::is&#95;optional&#95;impl</code>
</a>
</h3>

**Inherits From**:
`std::false_type`

<h3 id="struct-thrustdetailis-optional-impl<-optional<-t->->">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1is__optional__impl_3_01optional_3_01t_01_4_01_4.html">Struct <code>thrust::detail::is&#95;optional&#95;impl&lt; optional&lt; T &gt; &gt;</code>
</a>
</h3>

**Inherits From**:
`std::true_type`

<h3 id="struct-thrustdetailis-swappable">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1is__swappable.html">Struct <code>thrust::detail::is&#95;swappable</code>
</a>
</h3>

**Inherits From**:
`std::true_type`

<h3 id="struct-thrustdetailoptional-copy-assign-base">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__copy__assign__base.html">Struct <code>thrust::detail::optional&#95;copy&#95;assign&#95;base</code>
</a>
</h3>

**Inherits From**:
`thrust::detail::optional_move_base< T >`

**Inherited By**:
* [`thrust::detail::optional_move_assign_base< T, bool >`]({{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__move__assign__base.html)
* [`thrust::detail::optional_move_assign_base< T, false >`]({{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__move__assign__base_3_01t_00_01false_01_4.html)

<h3 id="struct-thrustdetailoptional-copy-assign-base<-t,-false->">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__copy__assign__base_3_01t_00_01false_01_4.html">Struct <code>thrust::detail::optional&#95;copy&#95;assign&#95;base&lt; T, false &gt;</code>
</a>
</h3>

**Inherits From**:
`thrust::detail::optional_move_base< T >`

<h3 id="struct-thrustdetailoptional-copy-base">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__copy__base.html">Struct <code>thrust::detail::optional&#95;copy&#95;base</code>
</a>
</h3>

**Inherits From**:
`thrust::detail::optional_operations_base< T >`

**Inherited By**:
* [`thrust::detail::optional_move_base< T, bool >`]({{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__move__base.html)
* [`thrust::detail::optional_move_base< T, false >`]({{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__move__base_3_01t_00_01false_01_4.html)

<h3 id="struct-thrustdetailoptional-copy-base<-t,-false->">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__copy__base_3_01t_00_01false_01_4.html">Struct <code>thrust::detail::optional&#95;copy&#95;base&lt; T, false &gt;</code>
</a>
</h3>

**Inherits From**:
`thrust::detail::optional_operations_base< T >`

<h3 id="struct-thrustdetailoptional-delete-assign-base">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__delete__assign__base.html">Struct <code>thrust::detail::optional&#95;delete&#95;assign&#95;base</code>
</a>
</h3>

**Inherited By**:
[`thrust::optional< T >`]({{ site.baseurl }}/api/classes/classthrust_1_1optional.html)

<h3 id="struct-thrustdetailoptional-delete-assign-base<-t,-false,-false->">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__delete__assign__base_3_01t_00_01false_00_01false_01_4.html">Struct <code>thrust::detail::optional&#95;delete&#95;assign&#95;base&lt; T, false, false &gt;</code>
</a>
</h3>

<h3 id="struct-thrustdetailoptional-delete-assign-base<-t,-false,-true->">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__delete__assign__base_3_01t_00_01false_00_01true_01_4.html">Struct <code>thrust::detail::optional&#95;delete&#95;assign&#95;base&lt; T, false, true &gt;</code>
</a>
</h3>

<h3 id="struct-thrustdetailoptional-delete-assign-base<-t,-true,-false->">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__delete__assign__base_3_01t_00_01true_00_01false_01_4.html">Struct <code>thrust::detail::optional&#95;delete&#95;assign&#95;base&lt; T, true, false &gt;</code>
</a>
</h3>

<h3 id="struct-thrustdetailoptional-delete-ctor-base">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__delete__ctor__base.html">Struct <code>thrust::detail::optional&#95;delete&#95;ctor&#95;base</code>
</a>
</h3>

**Inherited By**:
[`thrust::optional< T >`]({{ site.baseurl }}/api/classes/classthrust_1_1optional.html)

<h3 id="struct-thrustdetailoptional-delete-ctor-base<-t,-false,-false->">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__delete__ctor__base_3_01t_00_01false_00_01false_01_4.html">Struct <code>thrust::detail::optional&#95;delete&#95;ctor&#95;base&lt; T, false, false &gt;</code>
</a>
</h3>

<h3 id="struct-thrustdetailoptional-delete-ctor-base<-t,-false,-true->">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__delete__ctor__base_3_01t_00_01false_00_01true_01_4.html">Struct <code>thrust::detail::optional&#95;delete&#95;ctor&#95;base&lt; T, false, true &gt;</code>
</a>
</h3>

<h3 id="struct-thrustdetailoptional-delete-ctor-base<-t,-true,-false->">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__delete__ctor__base_3_01t_00_01true_00_01false_01_4.html">Struct <code>thrust::detail::optional&#95;delete&#95;ctor&#95;base&lt; T, true, false &gt;</code>
</a>
</h3>

<h3 id="struct-thrustdetailoptional-move-assign-base">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__move__assign__base.html">Struct <code>thrust::detail::optional&#95;move&#95;assign&#95;base</code>
</a>
</h3>

**Inherits From**:
`thrust::detail::optional_copy_assign_base< T >`

**Inherited By**:
[`thrust::optional< T >`]({{ site.baseurl }}/api/classes/classthrust_1_1optional.html)

<h3 id="struct-thrustdetailoptional-move-assign-base<-t,-false->">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__move__assign__base_3_01t_00_01false_01_4.html">Struct <code>thrust::detail::optional&#95;move&#95;assign&#95;base&lt; T, false &gt;</code>
</a>
</h3>

**Inherits From**:
`thrust::detail::optional_copy_assign_base< T >`

<h3 id="struct-thrustdetailoptional-move-base">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__move__base.html">Struct <code>thrust::detail::optional&#95;move&#95;base</code>
</a>
</h3>

**Inherits From**:
`thrust::detail::optional_copy_base< T >`

**Inherited By**:
* [`thrust::detail::optional_copy_assign_base< T, bool >`]({{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__copy__assign__base.html)
* [`thrust::detail::optional_copy_assign_base< T, false >`]({{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__copy__assign__base_3_01t_00_01false_01_4.html)

<h3 id="struct-thrustdetailoptional-move-base<-t,-false->">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__move__base_3_01t_00_01false_01_4.html">Struct <code>thrust::detail::optional&#95;move&#95;base&lt; T, false &gt;</code>
</a>
</h3>

**Inherits From**:
`thrust::detail::optional_copy_base< T >`

<h3 id="struct-thrustdetailoptional-operations-base">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__operations__base.html">Struct <code>thrust::detail::optional&#95;operations&#95;base</code>
</a>
</h3>

**Inherits From**:
`thrust::detail::optional_storage_base< T >`

**Inherited By**:
* [`thrust::detail::optional_copy_base< T, bool >`]({{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__copy__base.html)
* [`thrust::detail::optional_copy_base< T, false >`]({{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__copy__base_3_01t_00_01false_01_4.html)

<h3 id="struct-thrustdetailoptional-storage-base">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__storage__base.html">Struct <code>thrust::detail::optional&#95;storage&#95;base</code>
</a>
</h3>

**Inherited By**:
[`thrust::detail::optional_operations_base< T >`]({{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__operations__base.html)

<h3 id="struct-thrustdetailoptional-storage-base<-t,-true->">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1optional__storage__base_3_01t_00_01true_01_4.html">Struct <code>thrust::detail::optional&#95;storage&#95;base&lt; T, true &gt;</code>
</a>
</h3>

<h3 id="struct-thrustdetailreturns-void-impl">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1returns__void__impl.html">Struct <code>thrust::detail::returns&#95;void&#95;impl</code>
</a>
</h3>

<h3 id="struct-thrustdetailreturns-void-impl<-f,-void-t<-invoke-result-t<-f,-u...->->,-u...->">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1returns__void__impl_3_01f_00_01void__t_3_01invoke__result__t_3_01f_00_92d99556b5b7d67309a0911581ace58d.html">Struct <code>thrust::detail::returns&#95;void&#95;impl&lt; F, void&#95;t&lt; invoke&#95;result&#95;t&lt; F, U... &gt; &gt;, U... &gt;</code>
</a>
</h3>

**Inherits From**:
`std::is_void< invoke_result_t< F, U... > >`

<h3 id="struct-thrustdetailvoider">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1detail_1_1voider.html">Struct <code>thrust::detail::voider</code>
</a>
</h3>


## Types

<h3 id="using-remove-const-t">
Type Alias <code>thrust::detail::remove&#95;const&#95;t</code>
</h3>

<code class="doxybook">
<span>template &lt;class T&gt;</span>
<span>using <b>remove_const_t</b> = typename std::remove&#95;const&lt; T &gt;::type;</span></code>
<h3 id="using-remove-reference-t">
Type Alias <code>thrust::detail::remove&#95;reference&#95;t</code>
</h3>

<code class="doxybook">
<span>template &lt;class T&gt;</span>
<span>using <b>remove_reference_t</b> = typename std::remove&#95;reference&lt; T &gt;::type;</span></code>
<h3 id="using-decay-t">
Type Alias <code>thrust::detail::decay&#95;t</code>
</h3>

<code class="doxybook">
<span>template &lt;class T&gt;</span>
<span>using <b>decay_t</b> = typename std::decay&lt; T &gt;::type;</span></code>
<h3 id="using-enable-if-t">
Type Alias <code>thrust::detail::enable&#95;if&#95;t</code>
</h3>

<code class="doxybook">
<span>template &lt;bool E,</span>
<span>&nbsp;&nbsp;class T = void&gt;</span>
<span>using <b>enable_if_t</b> = typename std::enable&#95;if&lt; E, T &gt;::type;</span></code>
<h3 id="using-conditional-t">
Type Alias <code>thrust::detail::conditional&#95;t</code>
</h3>

<code class="doxybook">
<span>template &lt;bool B,</span>
<span>&nbsp;&nbsp;class T,</span>
<span>&nbsp;&nbsp;class F&gt;</span>
<span>using <b>conditional_t</b> = typename std::conditional&lt; B, T, F &gt;::type;</span></code>
<h3 id="using-void-t">
Type Alias <code>thrust::detail::void&#95;t</code>
</h3>

<code class="doxybook">
<span>template &lt;class... Ts&gt;</span>
<span>using <b>void_t</b> = typename voider&lt; Ts... &gt;::type;</span></code>
<h3 id="using-is-optional">
Type Alias <code>thrust::detail::is&#95;optional</code>
</h3>

<code class="doxybook">
<span>template &lt;class T&gt;</span>
<span>using <b>is_optional</b> = is&#95;optional&#95;impl&lt; decay&#95;t&lt; T &gt; &gt;;</span></code>
<h3 id="using-fixup-void">
Type Alias <code>thrust::detail::fixup&#95;void</code>
</h3>

<code class="doxybook">
<span>template &lt;class U&gt;</span>
<span>using <b>fixup_void</b> = conditional&#95;t&lt; std::is&#95;void&lt; U &gt;::value, &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1monostate.html"&gt;monostate&lt;/a&gt;, U &gt;;</span></code>
<h3 id="using-get-map-return">
Type Alias <code>thrust::detail::get&#95;map&#95;return</code>
</h3>

<code class="doxybook">
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class U,</span>
<span>&nbsp;&nbsp;class = invoke&#95;result&#95;t&lt;F, U&gt;&gt;</span>
<span>using <b>get_map_return</b> = &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1optional.html"&gt;optional&lt;/a&gt;&lt; fixup&#95;void&lt; invoke&#95;result&#95;t&lt; F, U &gt; &gt; &gt;;</span></code>
<h3 id="using-returns-void">
Type Alias <code>thrust::detail::returns&#95;void</code>
</h3>

<code class="doxybook">
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class... U&gt;</span>
<span>using <b>returns_void</b> = returns&#95;void&#95;impl&lt; F, void, U... &gt;;</span></code>
<h3 id="using-enable-if-ret-void">
Type Alias <code>thrust::detail::enable&#95;if&#95;ret&#95;void</code>
</h3>

<code class="doxybook">
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class... U&gt;</span>
<span>using <b>enable_if_ret_void</b> = enable&#95;if&#95;t&lt; returns&#95;void&lt; T &&, U... &gt;::value &gt;;</span></code>
<h3 id="using-disable-if-ret-void">
Type Alias <code>thrust::detail::disable&#95;if&#95;ret&#95;void</code>
</h3>

<code class="doxybook">
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class... U&gt;</span>
<span>using <b>disable_if_ret_void</b> = enable&#95;if&#95;t&lt;!returns&#95;void&lt; T &&, U... &gt;::value &gt;;</span></code>
<h3 id="using-enable-forward-value">
Type Alias <code>thrust::detail::enable&#95;forward&#95;value</code>
</h3>

<code class="doxybook">
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class U&gt;</span>
<span>using <b>enable_forward_value</b> = detail::enable&#95;if&#95;t&lt; std::is&#95;constructible&lt; T, U && &gt;::value &&!std::is&#95;same&lt; detail::decay&#95;t&lt; U &gt;, &lt;a href="{{ site.baseurl }}/api/classes/structthrust&#95;1&#95;1in&#95;&#95;place&#95;&#95;t.html"&gt;in&#95;place&#95;t&lt;/a&gt; &gt;::value &&!std::is&#95;same&lt; &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1optional.html"&gt;optional&lt;/a&gt;&lt; T &gt;, detail::decay&#95;t&lt; U &gt; &gt;::value &gt;;</span></code>
<h3 id="using-enable-from-other">
Type Alias <code>thrust::detail::enable&#95;from&#95;other</code>
</h3>

<code class="doxybook">
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class U,</span>
<span>&nbsp;&nbsp;class Other&gt;</span>
<span>using <b>enable_from_other</b> = detail::enable&#95;if&#95;t&lt; std::is&#95;constructible&lt; T, Other &gt;::value &&!std::is&#95;constructible&lt; T, &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1optional.html"&gt;optional&lt;/a&gt;&lt; U &gt; & &gt;::value &&!std::is&#95;constructible&lt; T, &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1optional.html"&gt;optional&lt;/a&gt;&lt; U &gt; && &gt;::value &&!std::is&#95;constructible&lt; T, const &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1optional.html"&gt;optional&lt;/a&gt;&lt; U &gt; & &gt;::value &&!std::is&#95;constructible&lt; T, const &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1optional.html"&gt;optional&lt;/a&gt;&lt; U &gt; && &gt;::value &&!std::is&#95;convertible&lt; &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1optional.html"&gt;optional&lt;/a&gt;&lt; U &gt; &, T &gt;::value &&!std::is&#95;convertible&lt; &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1optional.html"&gt;optional&lt;/a&gt;&lt; U &gt; &&, T &gt;::value &&!std::is&#95;convertible&lt; const &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1optional.html"&gt;optional&lt;/a&gt;&lt; U &gt; &, T &gt;::value &&!std::is&#95;convertible&lt; const &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1optional.html"&gt;optional&lt;/a&gt;&lt; U &gt; &&, T &gt;::value &gt;;</span></code>
<h3 id="using-enable-assign-forward">
Type Alias <code>thrust::detail::enable&#95;assign&#95;forward</code>
</h3>

<code class="doxybook">
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class U&gt;</span>
<span>using <b>enable_assign_forward</b> = detail::enable&#95;if&#95;t&lt; !std::is&#95;same&lt; &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1optional.html"&gt;optional&lt;/a&gt;&lt; T &gt;, detail::decay&#95;t&lt; U &gt; &gt;::value &&!detail::conjunction&lt; std::is&#95;scalar&lt; T &gt;, std::is&#95;same&lt; T, detail::decay&#95;t&lt; U &gt; &gt; &gt;::value &&std::is&#95;constructible&lt; T, U &gt;::value &&std::is&#95;assignable&lt; T &, U &gt;::value &gt;;</span></code>
<h3 id="using-enable-assign-from-other">
Type Alias <code>thrust::detail::enable&#95;assign&#95;from&#95;other</code>
</h3>

<code class="doxybook">
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class U,</span>
<span>&nbsp;&nbsp;class Other&gt;</span>
<span>using <b>enable_assign_from_other</b> = detail::enable&#95;if&#95;t&lt; std::is&#95;constructible&lt; T, Other &gt;::value &&std::is&#95;assignable&lt; T &, Other &gt;::value &&!std::is&#95;constructible&lt; T, &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1optional.html"&gt;optional&lt;/a&gt;&lt; U &gt; & &gt;::value &&!std::is&#95;constructible&lt; T, &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1optional.html"&gt;optional&lt;/a&gt;&lt; U &gt; && &gt;::value &&!std::is&#95;constructible&lt; T, const &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1optional.html"&gt;optional&lt;/a&gt;&lt; U &gt; & &gt;::value &&!std::is&#95;constructible&lt; T, const &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1optional.html"&gt;optional&lt;/a&gt;&lt; U &gt; && &gt;::value &&!std::is&#95;convertible&lt; &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1optional.html"&gt;optional&lt;/a&gt;&lt; U &gt; &, T &gt;::value &&!std::is&#95;convertible&lt; &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1optional.html"&gt;optional&lt;/a&gt;&lt; U &gt; &&, T &gt;::value &&!std::is&#95;convertible&lt; const &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1optional.html"&gt;optional&lt;/a&gt;&lt; U &gt; &, T &gt;::value &&!std::is&#95;convertible&lt; const &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1optional.html"&gt;optional&lt;/a&gt;&lt; U &gt; &&, T &gt;::value &&!std::is&#95;assignable&lt; T &, &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1optional.html"&gt;optional&lt;/a&gt;&lt; U &gt; & &gt;::value &&!std::is&#95;assignable&lt; T &, &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1optional.html"&gt;optional&lt;/a&gt;&lt; U &gt; && &gt;::value &&!std::is&#95;assignable&lt; T &, const &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1optional.html"&gt;optional&lt;/a&gt;&lt; U &gt; & &gt;::value &&!std::is&#95;assignable&lt; T &, const &lt;a href="{{ site.baseurl }}/api/classes/classthrust&#95;1&#95;1optional.html"&gt;optional&lt;/a&gt;&lt; U &gt; && &gt;::value &gt;;</span></code>

## Functions

<h3 id="function-allocator-delete-impl">
Function <code>thrust::detail::allocator&#95;delete&#95;impl</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Allocator,</span>
<span>&nbsp;&nbsp;typename Pointer&gt;</span>
<span>void </span><span><b>allocator_delete_impl</b>(Allocator const & alloc,</span>
<span>&nbsp;&nbsp;Pointer p,</span>
<span>&nbsp;&nbsp;std::false_type);</span></code>
<h3 id="function-allocator-delete-impl">
Function <code>thrust::detail::allocator&#95;delete&#95;impl</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Allocator,</span>
<span>&nbsp;&nbsp;typename Pointer&gt;</span>
<span>void </span><span><b>allocator_delete_impl</b>(Allocator const & alloc,</span>
<span>&nbsp;&nbsp;Pointer p,</span>
<span>&nbsp;&nbsp;std::true_type);</span></code>
<h3 id="function-array-allocator-delete-impl">
Function <code>thrust::detail::array&#95;allocator&#95;delete&#95;impl</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Allocator,</span>
<span>&nbsp;&nbsp;typename Pointer,</span>
<span>&nbsp;&nbsp;typename Size&gt;</span>
<span>void </span><span><b>array_allocator_delete_impl</b>(Allocator const & alloc,</span>
<span>&nbsp;&nbsp;Pointer p,</span>
<span>&nbsp;&nbsp;Size count,</span>
<span>&nbsp;&nbsp;std::false_type);</span></code>
<h3 id="function-array-allocator-delete-impl">
Function <code>thrust::detail::array&#95;allocator&#95;delete&#95;impl</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Allocator,</span>
<span>&nbsp;&nbsp;typename Pointer,</span>
<span>&nbsp;&nbsp;typename Size&gt;</span>
<span>void </span><span><b>array_allocator_delete_impl</b>(Allocator const & alloc,</span>
<span>&nbsp;&nbsp;Pointer p,</span>
<span>&nbsp;&nbsp;Size count,</span>
<span>&nbsp;&nbsp;std::true_type);</span></code>
<h3 id="function-invoke">
Function <code>thrust::detail::invoke</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Fn,</span>
<span>&nbsp;&nbsp;typename... Args,</span>
<span>&nbsp;&nbsp;typename = enable&#95;if&#95;t&lt;std::is&#95;member&#95;pointer&lt;decay&#95;t&lt;Fn&gt;&gt;::value&gt;,</span>
<span>&nbsp;&nbsp;int = 0&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ constexpr auto </span><span><b>invoke</b>(Fn && f,</span>
<span>&nbsp;&nbsp;Args &&... args);</span></code>
<h3 id="function-invoke">
Function <code>thrust::detail::invoke</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Fn,</span>
<span>&nbsp;&nbsp;typename... Args,</span>
<span>&nbsp;&nbsp;typename = enable&#95;if&#95;t&lt;!std::is&#95;member&#95;pointer&lt;decay&#95;t&lt;Fn&gt;&gt;::value&gt;&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ constexpr auto </span><span><b>invoke</b>(Fn && f,</span>
<span>&nbsp;&nbsp;Args &&... args);</span></code>

