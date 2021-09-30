---
title: detail
summary: \exclude 
parent: Function Object Adaptors
grand_parent: Function Objects
nav_exclude: true
has_children: true
has_toc: false
---

# Namespace `detail`

<code class="doxybook">
<span>namespace detail {</span>
<br>
<span>namespace <b><a href="/thrust/api/namespaces/namespacedetail_1_1zip__detail.html">detail::zip&#95;detail</a></b> { <i>…</i> }</span>
<br>
<span>template &lt;class...&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1conjunction.html">conjunction</a></b>;</span>
<br>
<span>template &lt;class B&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1conjunction_3_01b_01_4.html">conjunction&lt; B &gt;</a></b>;</span>
<br>
<span>template &lt;class B,</span>
<span>&nbsp;&nbsp;class... Bs&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1conjunction_3_01b_00_01bs_8_8_8_01_4.html">conjunction&lt; B, Bs... &gt;</a></b>;</span>
<br>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1i__am__secret.html">i&#95;am&#95;secret</a></b>;</span>
<br>
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class,</span>
<span>&nbsp;&nbsp;class... Us&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1invoke__result__impl.html">invoke&#95;result&#95;impl</a></b>;</span>
<br>
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class... Us&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1invoke__result__impl_3_01f_00_01decltype_07detail_1_1invoke_07std_1_1declval_3_07c1f4caaa1db079732d0d4c8ba802ae7.html">invoke&#95;result&#95;impl&lt; F, decltype(detail::invoke(std::declval&lt; F &gt;(), std::declval&lt; Us &gt;()...), void()), Us... &gt;</a></b>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class U = T&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1is__nothrow__swappable.html">is&#95;nothrow&#95;swappable</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1is__optional__impl.html">is&#95;optional&#95;impl</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1is__optional__impl_3_01optional_3_01t_01_4_01_4.html">is&#95;optional&#95;impl&lt; optional&lt; T &gt; &gt;</a></b>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class U = T&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1is__swappable.html">is&#95;swappable</a></b>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;bool = std::is&#95;trivially&#95;copy&#95;assignable&lt; T &gt;::value && std::is&#95;trivially&#95;copy&#95;constructible&lt; T &gt;::value && std::is&#95;trivially&#95;destructible&lt; T &gt;::value&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1optional__copy__assign__base.html">optional&#95;copy&#95;assign&#95;base</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1optional__copy__assign__base_3_01t_00_01false_01_4.html">optional&#95;copy&#95;assign&#95;base&lt; T, false &gt;</a></b>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;bool = std::is&#95;trivially&#95;copy&#95;constructible&lt; T &gt;::value&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1optional__copy__base.html">optional&#95;copy&#95;base</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1optional__copy__base_3_01t_00_01false_01_4.html">optional&#95;copy&#95;base&lt; T, false &gt;</a></b>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;bool EnableCopy = (std::is&#95;copy&#95;constructible&lt;T&gt;::value &&                             std::is&#95;copy&#95;assignable&lt;T&gt;::value),</span>
<span>&nbsp;&nbsp;bool EnableMove = (std::is&#95;move&#95;constructible&lt;T&gt;::value &&                             std::is&#95;move&#95;assignable&lt;T&gt;::value)&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1optional__delete__assign__base.html">optional&#95;delete&#95;assign&#95;base</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1optional__delete__assign__base_3_01t_00_01false_00_01false_01_4.html">optional&#95;delete&#95;assign&#95;base&lt; T, false, false &gt;</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1optional__delete__assign__base_3_01t_00_01false_00_01true_01_4.html">optional&#95;delete&#95;assign&#95;base&lt; T, false, true &gt;</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1optional__delete__assign__base_3_01t_00_01true_00_01false_01_4.html">optional&#95;delete&#95;assign&#95;base&lt; T, true, false &gt;</a></b>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;bool EnableCopy = std::is&#95;copy&#95;constructible&lt;T&gt;::value,</span>
<span>&nbsp;&nbsp;bool EnableMove = std::is&#95;move&#95;constructible&lt;T&gt;::value&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1optional__delete__ctor__base.html">optional&#95;delete&#95;ctor&#95;base</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1optional__delete__ctor__base_3_01t_00_01false_00_01false_01_4.html">optional&#95;delete&#95;ctor&#95;base&lt; T, false, false &gt;</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1optional__delete__ctor__base_3_01t_00_01false_00_01true_01_4.html">optional&#95;delete&#95;ctor&#95;base&lt; T, false, true &gt;</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1optional__delete__ctor__base_3_01t_00_01true_00_01false_01_4.html">optional&#95;delete&#95;ctor&#95;base&lt; T, true, false &gt;</a></b>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;bool = std::is&#95;trivially&#95;destructible&lt; T &gt;::value && std::is&#95;trivially&#95;move&#95;constructible&lt; T &gt;::value && std::is&#95;trivially&#95;move&#95;assignable&lt; T &gt;::value&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1optional__move__assign__base.html">optional&#95;move&#95;assign&#95;base</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1optional__move__assign__base_3_01t_00_01false_01_4.html">optional&#95;move&#95;assign&#95;base&lt; T, false &gt;</a></b>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;bool = std::is&#95;trivially&#95;move&#95;constructible&lt; T &gt;::value&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1optional__move__base.html">optional&#95;move&#95;base</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1optional__move__base_3_01t_00_01false_01_4.html">optional&#95;move&#95;base&lt; T, false &gt;</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1optional__operations__base.html">optional&#95;operations&#95;base</a></b>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;bool = ::std::is&#95;trivially&#95;destructible&lt;T&gt;::value&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1optional__storage__base.html">optional&#95;storage&#95;base</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1optional__storage__base_3_01t_00_01true_01_4.html">optional&#95;storage&#95;base&lt; T, true &gt;</a></b>;</span>
<br>
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class = void,</span>
<span>&nbsp;&nbsp;class... U&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1returns__void__impl.html">returns&#95;void&#95;impl</a></b>;</span>
<br>
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class... U&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1returns__void__impl_3_01f_00_01void__t_3_01invoke__result__t_3_01f_00_01u_8_8_8_01_4_01_4_00_01u_8_8_8_01_4.html">returns&#95;void&#95;impl&lt; F, void&#95;t&lt; invoke&#95;result&#95;t&lt; F, U... &gt; &gt;, U... &gt;</a></b>;</span>
<br>
<span>template &lt;class...&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdetail_1_1voider.html">voider</a></b>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>using <b><a href="/thrust/api/namespaces/namespacedetail.html#using-remove_const_t">remove&#95;const&#95;t</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>using <b><a href="/thrust/api/namespaces/namespacedetail.html#using-remove_reference_t">remove&#95;reference&#95;t</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>using <b><a href="/thrust/api/namespaces/namespacedetail.html#using-decay_t">decay&#95;t</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;bool E,</span>
<span>&nbsp;&nbsp;class T = void&gt;</span>
<span>using <b><a href="/thrust/api/namespaces/namespacedetail.html#using-enable_if_t">enable&#95;if&#95;t</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;bool B,</span>
<span>&nbsp;&nbsp;class T,</span>
<span>&nbsp;&nbsp;class F&gt;</span>
<span>using <b><a href="/thrust/api/namespaces/namespacedetail.html#using-conditional_t">conditional&#95;t</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class... Us&gt;</span>
<span>using <b><a href="/thrust/api/namespaces/namespacedetail.html#using-invoke_result">invoke&#95;result</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class... Us&gt;</span>
<span>using <b><a href="/thrust/api/namespaces/namespacedetail.html#using-invoke_result_t">invoke&#95;result&#95;t</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class... Ts&gt;</span>
<span>using <b><a href="/thrust/api/namespaces/namespacedetail.html#using-void_t">void&#95;t</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class T&gt;</span>
<span>using <b><a href="/thrust/api/namespaces/namespacedetail.html#using-is_optional">is&#95;optional</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class U&gt;</span>
<span>using <b><a href="/thrust/api/namespaces/namespacedetail.html#using-fixup_void">fixup&#95;void</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class U,</span>
<span>&nbsp;&nbsp;class = invoke&#95;result&#95;t&lt;F, U&gt;&gt;</span>
<span>using <b><a href="/thrust/api/namespaces/namespacedetail.html#using-get_map_return">get&#95;map&#95;return</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class... U&gt;</span>
<span>using <b><a href="/thrust/api/namespaces/namespacedetail.html#using-returns_void">returns&#95;void</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class... U&gt;</span>
<span>using <b><a href="/thrust/api/namespaces/namespacedetail.html#using-enable_if_ret_void">enable&#95;if&#95;ret&#95;void</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class... U&gt;</span>
<span>using <b><a href="/thrust/api/namespaces/namespacedetail.html#using-disable_if_ret_void">disable&#95;if&#95;ret&#95;void</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class U&gt;</span>
<span>using <b><a href="/thrust/api/namespaces/namespacedetail.html#using-enable_forward_value">enable&#95;forward&#95;value</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class U,</span>
<span>&nbsp;&nbsp;class Other&gt;</span>
<span>using <b><a href="/thrust/api/namespaces/namespacedetail.html#using-enable_from_other">enable&#95;from&#95;other</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class U&gt;</span>
<span>using <b><a href="/thrust/api/namespaces/namespacedetail.html#using-enable_assign_forward">enable&#95;assign&#95;forward</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class U,</span>
<span>&nbsp;&nbsp;class Other&gt;</span>
<span>using <b><a href="/thrust/api/namespaces/namespacedetail.html#using-enable_assign_from_other">enable&#95;assign&#95;from&#95;other</a></b> = <i>see below</i>;</span>
<br>
<span>template &lt;typename Allocator,</span>
<span>&nbsp;&nbsp;typename Pointer&gt;</span>
<span>void </span><span><b><a href="/thrust/api/namespaces/namespacedetail.html#function-allocator_delete_impl">allocator&#95;delete&#95;impl</a></b>(Allocator const & alloc,</span>
<span>&nbsp;&nbsp;Pointer p,</span>
<span>&nbsp;&nbsp;std::false_type);</span>
<br>
<span>template &lt;typename Allocator,</span>
<span>&nbsp;&nbsp;typename Pointer&gt;</span>
<span>void </span><span><b><a href="/thrust/api/namespaces/namespacedetail.html#function-allocator_delete_impl">allocator&#95;delete&#95;impl</a></b>(Allocator const & alloc,</span>
<span>&nbsp;&nbsp;Pointer p,</span>
<span>&nbsp;&nbsp;std::true_type);</span>
<br>
<span>template &lt;typename Allocator,</span>
<span>&nbsp;&nbsp;typename Pointer,</span>
<span>&nbsp;&nbsp;typename Size&gt;</span>
<span>void </span><span><b><a href="/thrust/api/namespaces/namespacedetail.html#function-array_allocator_delete_impl">array&#95;allocator&#95;delete&#95;impl</a></b>(Allocator const & alloc,</span>
<span>&nbsp;&nbsp;Pointer p,</span>
<span>&nbsp;&nbsp;Size count,</span>
<span>&nbsp;&nbsp;std::false_type);</span>
<br>
<span>template &lt;typename Allocator,</span>
<span>&nbsp;&nbsp;typename Pointer,</span>
<span>&nbsp;&nbsp;typename Size&gt;</span>
<span>void </span><span><b><a href="/thrust/api/namespaces/namespacedetail.html#function-array_allocator_delete_impl">array&#95;allocator&#95;delete&#95;impl</a></b>(Allocator const & alloc,</span>
<span>&nbsp;&nbsp;Pointer p,</span>
<span>&nbsp;&nbsp;Size count,</span>
<span>&nbsp;&nbsp;std::true_type);</span>
<br>
<span>template &lt;typename Fn,</span>
<span>&nbsp;&nbsp;typename... Args,</span>
<span>&nbsp;&nbsp;typename = enable&#95;if&#95;t&lt;std::is&#95;member&#95;pointer&lt;decay&#95;t&lt;Fn&gt;&gt;::value&gt;,</span>
<span>&nbsp;&nbsp;int = 0&gt;</span>
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ auto </span><span><b><a href="/thrust/api/namespaces/namespacedetail.html#function-invoke">invoke</a></b>(Fn && f,</span>
<span>&nbsp;&nbsp;Args &&... args);</span>
<br>
<span>template &lt;typename Fn,</span>
<span>&nbsp;&nbsp;typename... Args,</span>
<span>&nbsp;&nbsp;typename = enable&#95;if&#95;t&lt;!std::is&#95;member&#95;pointer&lt;decay&#95;t&lt;Fn&gt;&gt;::value&gt;&gt;</span>
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ auto </span><span><b><a href="/thrust/api/namespaces/namespacedetail.html#function-invoke">invoke</a></b>(Fn && f,</span>
<span>&nbsp;&nbsp;Args &&... args);</span>
<br>
<span>template &lt;class Opt,</span>
<span>&nbsp;&nbsp;class F,</span>
<span>&nbsp;&nbsp;class Ret = decltype(detail::invoke(std::declval&lt;F&gt;(),                                              &#42;std::declval&lt;Opt&gt;())),</span>
<span>&nbsp;&nbsp;detail::enable_if_t<!std::is_void< Ret >::value > * = nullptr&gt;</span>
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ auto </span><span><b><a href="/thrust/api/namespaces/namespacedetail.html#function-optional_map_impl">optional&#95;map&#95;impl</a></b>(Opt && opt,</span>
<span>&nbsp;&nbsp;F && f);</span>
<br>
<span>template &lt;class Opt,</span>
<span>&nbsp;&nbsp;class F,</span>
<span>&nbsp;&nbsp;class Ret = decltype(detail::invoke(std::declval&lt;F&gt;(),                                              &#42;std::declval&lt;Opt&gt;())),</span>
<span>&nbsp;&nbsp;detail::enable_if_t< std::is_void< Ret >::value > * = nullptr&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ auto </span><span><b><a href="/thrust/api/namespaces/namespacedetail.html#function-optional_map_impl">optional&#95;map&#95;impl</a></b>(Opt && opt,</span>
<span>&nbsp;&nbsp;F && f);</span>
<span>} /* namespace detail */</span>
</code>

## Member Classes

<h3 id="struct-detail::conjunction">
<a href="/thrust/api/classes/structdetail_1_1conjunction.html">Struct <code>detail::conjunction</code>
</a>
</h3>

**Inherits From**:
`true_type`

<h3 id="struct-detail::conjunction< B >">
<a href="/thrust/api/classes/structdetail_1_1conjunction_3_01b_01_4.html">Struct <code>detail::conjunction&lt; B &gt;</code>
</a>
</h3>

**Inherits From**:
`B`

<h3 id="struct-detail::conjunction< B, Bs... >">
<a href="/thrust/api/classes/structdetail_1_1conjunction_3_01b_00_01bs_8_8_8_01_4.html">Struct <code>detail::conjunction&lt; B, Bs... &gt;</code>
</a>
</h3>

**Inherits From**:
`std::conditional::type< bool(B::value), conjunction< Bs... >, B >`

<h3 id="struct-detail::i_am_secret">
<a href="/thrust/api/classes/structdetail_1_1i__am__secret.html">Struct <code>detail::i&#95;am&#95;secret</code>
</a>
</h3>

<h3 id="struct-detail::invoke_result_impl">
<a href="/thrust/api/classes/structdetail_1_1invoke__result__impl.html">Struct <code>detail::invoke&#95;result&#95;impl</code>
</a>
</h3>

<h3 id="struct-detail::invoke_result_impl< F, decltype(detail::invoke(std::declval< F >(), std::declval< Us >()...), void()), Us... >">
<a href="/thrust/api/classes/structdetail_1_1invoke__result__impl_3_01f_00_01decltype_07detail_1_1invoke_07std_1_1declval_3_07c1f4caaa1db079732d0d4c8ba802ae7.html">Struct <code>detail::invoke&#95;result&#95;impl&lt; F, decltype(detail::invoke(std::declval&lt; F &gt;(), std::declval&lt; Us &gt;()...), void()), Us... &gt;</code>
</a>
</h3>

<h3 id="struct-detail::is_nothrow_swappable">
<a href="/thrust/api/classes/structdetail_1_1is__nothrow__swappable.html">Struct <code>detail::is&#95;nothrow&#95;swappable</code>
</a>
</h3>

**Inherits From**:
`true_type`

<h3 id="struct-detail::is_optional_impl">
<a href="/thrust/api/classes/structdetail_1_1is__optional__impl.html">Struct <code>detail::is&#95;optional&#95;impl</code>
</a>
</h3>

**Inherits From**:
`false_type`

<h3 id="struct-detail::is_optional_impl< optional< T > >">
<a href="/thrust/api/classes/structdetail_1_1is__optional__impl_3_01optional_3_01t_01_4_01_4.html">Struct <code>detail::is&#95;optional&#95;impl&lt; optional&lt; T &gt; &gt;</code>
</a>
</h3>

**Inherits From**:
`true_type`

<h3 id="struct-detail::is_swappable">
<a href="/thrust/api/classes/structdetail_1_1is__swappable.html">Struct <code>detail::is&#95;swappable</code>
</a>
</h3>

**Inherits From**:
`true_type`

<h3 id="struct-detail::optional_copy_assign_base">
<a href="/thrust/api/classes/structdetail_1_1optional__copy__assign__base.html">Struct <code>detail::optional&#95;copy&#95;assign&#95;base</code>
</a>
</h3>

**Inherits From**:
`detail::optional_move_base< T >`

**Inherited By**:
* [`detail::optional_move_assign_base< T, bool >`](/thrust/api/classes/structdetail_1_1optional__move__assign__base.html)
* [`detail::optional_move_assign_base< T, false >`](/thrust/api/classes/structdetail_1_1optional__move__assign__base_3_01t_00_01false_01_4.html)

<h3 id="struct-detail::optional_copy_assign_base< T, false >">
<a href="/thrust/api/classes/structdetail_1_1optional__copy__assign__base_3_01t_00_01false_01_4.html">Struct <code>detail::optional&#95;copy&#95;assign&#95;base&lt; T, false &gt;</code>
</a>
</h3>

**Inherits From**:
`detail::optional_move_base< T >`

<h3 id="struct-detail::optional_copy_base">
<a href="/thrust/api/classes/structdetail_1_1optional__copy__base.html">Struct <code>detail::optional&#95;copy&#95;base</code>
</a>
</h3>

**Inherits From**:
`detail::optional_operations_base< T >`

**Inherited By**:
* [`detail::optional_move_base< T, bool >`](/thrust/api/classes/structdetail_1_1optional__move__base.html)
* [`detail::optional_move_base< T, false >`](/thrust/api/classes/structdetail_1_1optional__move__base_3_01t_00_01false_01_4.html)
* [`detail::optional_move_base< T >`](/thrust/api/classes/structdetail_1_1optional__move__base.html)

<h3 id="struct-detail::optional_copy_base< T, false >">
<a href="/thrust/api/classes/structdetail_1_1optional__copy__base_3_01t_00_01false_01_4.html">Struct <code>detail::optional&#95;copy&#95;base&lt; T, false &gt;</code>
</a>
</h3>

**Inherits From**:
`detail::optional_operations_base< T >`

<h3 id="struct-detail::optional_delete_assign_base">
<a href="/thrust/api/classes/structdetail_1_1optional__delete__assign__base.html">Struct <code>detail::optional&#95;delete&#95;assign&#95;base</code>
</a>
</h3>

<h3 id="struct-detail::optional_delete_assign_base< T, false, false >">
<a href="/thrust/api/classes/structdetail_1_1optional__delete__assign__base_3_01t_00_01false_00_01false_01_4.html">Struct <code>detail::optional&#95;delete&#95;assign&#95;base&lt; T, false, false &gt;</code>
</a>
</h3>

<h3 id="struct-detail::optional_delete_assign_base< T, false, true >">
<a href="/thrust/api/classes/structdetail_1_1optional__delete__assign__base_3_01t_00_01false_00_01true_01_4.html">Struct <code>detail::optional&#95;delete&#95;assign&#95;base&lt; T, false, true &gt;</code>
</a>
</h3>

<h3 id="struct-detail::optional_delete_assign_base< T, true, false >">
<a href="/thrust/api/classes/structdetail_1_1optional__delete__assign__base_3_01t_00_01true_00_01false_01_4.html">Struct <code>detail::optional&#95;delete&#95;assign&#95;base&lt; T, true, false &gt;</code>
</a>
</h3>

<h3 id="struct-detail::optional_delete_ctor_base">
<a href="/thrust/api/classes/structdetail_1_1optional__delete__ctor__base.html">Struct <code>detail::optional&#95;delete&#95;ctor&#95;base</code>
</a>
</h3>

<h3 id="struct-detail::optional_delete_ctor_base< T, false, false >">
<a href="/thrust/api/classes/structdetail_1_1optional__delete__ctor__base_3_01t_00_01false_00_01false_01_4.html">Struct <code>detail::optional&#95;delete&#95;ctor&#95;base&lt; T, false, false &gt;</code>
</a>
</h3>

<h3 id="struct-detail::optional_delete_ctor_base< T, false, true >">
<a href="/thrust/api/classes/structdetail_1_1optional__delete__ctor__base_3_01t_00_01false_00_01true_01_4.html">Struct <code>detail::optional&#95;delete&#95;ctor&#95;base&lt; T, false, true &gt;</code>
</a>
</h3>

<h3 id="struct-detail::optional_delete_ctor_base< T, true, false >">
<a href="/thrust/api/classes/structdetail_1_1optional__delete__ctor__base_3_01t_00_01true_00_01false_01_4.html">Struct <code>detail::optional&#95;delete&#95;ctor&#95;base&lt; T, true, false &gt;</code>
</a>
</h3>

<h3 id="struct-detail::optional_move_assign_base">
<a href="/thrust/api/classes/structdetail_1_1optional__move__assign__base.html">Struct <code>detail::optional&#95;move&#95;assign&#95;base</code>
</a>
</h3>

**Inherits From**:
`detail::optional_copy_assign_base< T >`

<h3 id="struct-detail::optional_move_assign_base< T, false >">
<a href="/thrust/api/classes/structdetail_1_1optional__move__assign__base_3_01t_00_01false_01_4.html">Struct <code>detail::optional&#95;move&#95;assign&#95;base&lt; T, false &gt;</code>
</a>
</h3>

**Inherits From**:
`detail::optional_copy_assign_base< T >`

<h3 id="struct-detail::optional_move_base">
<a href="/thrust/api/classes/structdetail_1_1optional__move__base.html">Struct <code>detail::optional&#95;move&#95;base</code>
</a>
</h3>

**Inherits From**:
`detail::optional_copy_base< T >`

**Inherited By**:
* [`detail::optional_copy_assign_base< T, bool >`](/thrust/api/classes/structdetail_1_1optional__copy__assign__base.html)
* [`detail::optional_copy_assign_base< T, false >`](/thrust/api/classes/structdetail_1_1optional__copy__assign__base_3_01t_00_01false_01_4.html)
* [`detail::optional_copy_assign_base< T >`](/thrust/api/classes/structdetail_1_1optional__copy__assign__base.html)

<h3 id="struct-detail::optional_move_base< T, false >">
<a href="/thrust/api/classes/structdetail_1_1optional__move__base_3_01t_00_01false_01_4.html">Struct <code>detail::optional&#95;move&#95;base&lt; T, false &gt;</code>
</a>
</h3>

**Inherits From**:
`detail::optional_copy_base< T >`

<h3 id="struct-detail::optional_operations_base">
<a href="/thrust/api/classes/structdetail_1_1optional__operations__base.html">Struct <code>detail::optional&#95;operations&#95;base</code>
</a>
</h3>

**Inherits From**:
`detail::optional_storage_base< T >`

**Inherited By**:
* [`detail::optional_copy_base< T, bool >`](/thrust/api/classes/structdetail_1_1optional__copy__base.html)
* [`detail::optional_copy_base< T, false >`](/thrust/api/classes/structdetail_1_1optional__copy__base_3_01t_00_01false_01_4.html)
* [`detail::optional_copy_base< T >`](/thrust/api/classes/structdetail_1_1optional__copy__base.html)

<h3 id="struct-detail::optional_storage_base">
<a href="/thrust/api/classes/structdetail_1_1optional__storage__base.html">Struct <code>detail::optional&#95;storage&#95;base</code>
</a>
</h3>

**Inherited By**:
[`detail::optional_operations_base< T >`](/thrust/api/classes/structdetail_1_1optional__operations__base.html)

<h3 id="struct-detail::optional_storage_base< T, true >">
<a href="/thrust/api/classes/structdetail_1_1optional__storage__base_3_01t_00_01true_01_4.html">Struct <code>detail::optional&#95;storage&#95;base&lt; T, true &gt;</code>
</a>
</h3>

<h3 id="struct-detail::returns_void_impl">
<a href="/thrust/api/classes/structdetail_1_1returns__void__impl.html">Struct <code>detail::returns&#95;void&#95;impl</code>
</a>
</h3>

<h3 id="struct-detail::returns_void_impl< F, void_t< invoke_result_t< F, U... > >, U... >">
<a href="/thrust/api/classes/structdetail_1_1returns__void__impl_3_01f_00_01void__t_3_01invoke__result__t_3_01f_00_01u_8_8_8_01_4_01_4_00_01u_8_8_8_01_4.html">Struct <code>detail::returns&#95;void&#95;impl&lt; F, void&#95;t&lt; invoke&#95;result&#95;t&lt; F, U... &gt; &gt;, U... &gt;</code>
</a>
</h3>

**Inherits From**:
`std::is_void< invoke_result_t< F, U... > >`

<h3 id="struct-detail::voider">
<a href="/thrust/api/classes/structdetail_1_1voider.html">Struct <code>detail::voider</code>
</a>
</h3>


## Types

<h3 id="using-remove_const_t">
Type Alias <code>remove&#95;const&#95;t</code>
</h3>

<code class="doxybook">
<span>template &lt;class T&gt;</span>
<span>using <b>remove_const_t</b> = typename std::remove&#95;const&lt; T &gt;::type;</span></code>
<h3 id="using-remove_reference_t">
Type Alias <code>remove&#95;reference&#95;t</code>
</h3>

<code class="doxybook">
<span>template &lt;class T&gt;</span>
<span>using <b>remove_reference_t</b> = typename std::remove&#95;reference&lt; T &gt;::type;</span></code>
<h3 id="using-decay_t">
Type Alias <code>decay&#95;t</code>
</h3>

<code class="doxybook">
<span>template &lt;class T&gt;</span>
<span>using <b>decay_t</b> = typename std::decay&lt; T &gt;::type;</span></code>
<h3 id="using-enable_if_t">
Type Alias <code>enable&#95;if&#95;t</code>
</h3>

<code class="doxybook">
<span>template &lt;bool E,</span>
<span>&nbsp;&nbsp;class T = void&gt;</span>
<span>using <b>enable_if_t</b> = typename std::enable&#95;if&lt; E, T &gt;::type;</span></code>
<h3 id="using-conditional_t">
Type Alias <code>conditional&#95;t</code>
</h3>

<code class="doxybook">
<span>template &lt;bool B,</span>
<span>&nbsp;&nbsp;class T,</span>
<span>&nbsp;&nbsp;class F&gt;</span>
<span>using <b>conditional_t</b> = typename std::conditional&lt; B, T, F &gt;::type;</span></code>
<h3 id="using-invoke_result">
Type Alias <code>invoke&#95;result</code>
</h3>

<code class="doxybook">
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class... Us&gt;</span>
<span>using <b>invoke_result</b> = invoke&#95;result&#95;impl&lt; F, void, Us... &gt;;</span></code>
<h3 id="using-invoke_result_t">
Type Alias <code>invoke&#95;result&#95;t</code>
</h3>

<code class="doxybook">
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class... Us&gt;</span>
<span>using <b>invoke_result_t</b> = typename invoke&#95;result&lt; F, Us... &gt;::type;</span></code>
<h3 id="using-void_t">
Type Alias <code>void&#95;t</code>
</h3>

<code class="doxybook">
<span>template &lt;class... Ts&gt;</span>
<span>using <b>void_t</b> = typename voider&lt; Ts... &gt;::type;</span></code>
<h3 id="using-is_optional">
Type Alias <code>is&#95;optional</code>
</h3>

<code class="doxybook">
<span>template &lt;class T&gt;</span>
<span>using <b>is_optional</b> = is&#95;optional&#95;impl&lt; decay&#95;t&lt; T &gt; &gt;;</span></code>
<h3 id="using-fixup_void">
Type Alias <code>fixup&#95;void</code>
</h3>

<code class="doxybook">
<span>template &lt;class U&gt;</span>
<span>using <b>fixup_void</b> = conditional&#95;t&lt; std::is&#95;void&lt; U &gt;::value, &lt;a href="/thrust/api/classes/classmonostate.html"&gt;monostate&lt;/a&gt;, U &gt;;</span></code>
<h3 id="using-get_map_return">
Type Alias <code>get&#95;map&#95;return</code>
</h3>

<code class="doxybook">
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class U,</span>
<span>&nbsp;&nbsp;class = invoke&#95;result&#95;t&lt;F, U&gt;&gt;</span>
<span>using <b>get_map_return</b> = &lt;a href="/thrust/api/classes/classoptional.html"&gt;optional&lt;/a&gt;&lt; fixup&#95;void&lt; invoke&#95;result&#95;t&lt; F, U &gt; &gt;&gt;;</span></code>
<h3 id="using-returns_void">
Type Alias <code>returns&#95;void</code>
</h3>

<code class="doxybook">
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class... U&gt;</span>
<span>using <b>returns_void</b> = returns&#95;void&#95;impl&lt; F, void, U... &gt;;</span></code>
<h3 id="using-enable_if_ret_void">
Type Alias <code>enable&#95;if&#95;ret&#95;void</code>
</h3>

<code class="doxybook">
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class... U&gt;</span>
<span>using <b>enable_if_ret_void</b> = enable&#95;if&#95;t&lt; returns&#95;void&lt; T &&, U... &gt;::value &gt;;</span></code>
<h3 id="using-disable_if_ret_void">
Type Alias <code>disable&#95;if&#95;ret&#95;void</code>
</h3>

<code class="doxybook">
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class... U&gt;</span>
<span>using <b>disable_if_ret_void</b> = enable&#95;if&#95;t&lt;!returns&#95;void&lt; T &&, U... &gt;::value &gt;;</span></code>
<h3 id="using-enable_forward_value">
Type Alias <code>enable&#95;forward&#95;value</code>
</h3>

<code class="doxybook">
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class U&gt;</span>
<span>using <b>enable_forward_value</b> = detail::enable&#95;if&#95;t&lt; std::is&#95;constructible&lt; T, U && &gt;::value &&!std::is&#95;same&lt; detail::decay&#95;t&lt; U &gt;, &lt;a href="/thrust/api/classes/structin&#95;&#95;place&#95;&#95;t.html"&gt;in&#95;place&#95;t&lt;/a&gt; &gt;::value &&!std::is&#95;same&lt; &lt;a href="/thrust/api/classes/classoptional.html"&gt;optional&lt;/a&gt;&lt; T &gt;, detail::decay&#95;t&lt; U &gt; &gt;::value &gt;;</span></code>
<h3 id="using-enable_from_other">
Type Alias <code>enable&#95;from&#95;other</code>
</h3>

<code class="doxybook">
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class U,</span>
<span>&nbsp;&nbsp;class Other&gt;</span>
<span>using <b>enable_from_other</b> = detail::enable&#95;if&#95;t&lt; std::is&#95;constructible&lt; T, Other &gt;::value &&!std::is&#95;constructible&lt; T, &lt;a href="/thrust/api/classes/classoptional.html"&gt;optional&lt;/a&gt;&lt; U &gt; & &gt;::value &&!std::is&#95;constructible&lt; T, &lt;a href="/thrust/api/classes/classoptional.html"&gt;optional&lt;/a&gt;&lt; U &gt; && &gt;::value &&!std::is&#95;constructible&lt; T, const &lt;a href="/thrust/api/classes/classoptional.html"&gt;optional&lt;/a&gt;&lt; U &gt; & &gt;::value &&!std::is&#95;constructible&lt; T, const &lt;a href="/thrust/api/classes/classoptional.html"&gt;optional&lt;/a&gt;&lt; U &gt; && &gt;::value &&!std::is&#95;convertible&lt; &lt;a href="/thrust/api/classes/classoptional.html"&gt;optional&lt;/a&gt;&lt; U &gt; &, T &gt;::value &&!std::is&#95;convertible&lt; &lt;a href="/thrust/api/classes/classoptional.html"&gt;optional&lt;/a&gt;&lt; U &gt; &&, T &gt;::value &&!std::is&#95;convertible&lt; const &lt;a href="/thrust/api/classes/classoptional.html"&gt;optional&lt;/a&gt;&lt; U &gt; &, T &gt;::value &&!std::is&#95;convertible&lt; const &lt;a href="/thrust/api/classes/classoptional.html"&gt;optional&lt;/a&gt;&lt; U &gt; &&, T &gt;::value &gt;;</span></code>
<h3 id="using-enable_assign_forward">
Type Alias <code>enable&#95;assign&#95;forward</code>
</h3>

<code class="doxybook">
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class U&gt;</span>
<span>using <b>enable_assign_forward</b> = detail::enable&#95;if&#95;t&lt; !std::is&#95;same&lt; &lt;a href="/thrust/api/classes/classoptional.html"&gt;optional&lt;/a&gt;&lt; T &gt;, detail::decay&#95;t&lt; U &gt; &gt;::value &&!&lt;a href="/thrust/api/groups/group&#95;&#95;type&#95;&#95;traits.html#using-conjunction"&gt;detail::conjunction&lt;/a&gt;&lt; std::is&#95;scalar&lt; T &gt;, std::is&#95;same&lt; T, detail::decay&#95;t&lt; U &gt; &gt;&gt;::value &&std::is&#95;constructible&lt; T, U &gt;::value &&std::is&#95;assignable&lt; T &, U &gt;::value &gt;;</span></code>
<h3 id="using-enable_assign_from_other">
Type Alias <code>enable&#95;assign&#95;from&#95;other</code>
</h3>

<code class="doxybook">
<span>template &lt;class T,</span>
<span>&nbsp;&nbsp;class U,</span>
<span>&nbsp;&nbsp;class Other&gt;</span>
<span>using <b>enable_assign_from_other</b> = detail::enable&#95;if&#95;t&lt; std::is&#95;constructible&lt; T, Other &gt;::value &&std::is&#95;assignable&lt; T &, Other &gt;::value &&!std::is&#95;constructible&lt; T, &lt;a href="/thrust/api/classes/classoptional.html"&gt;optional&lt;/a&gt;&lt; U &gt; & &gt;::value &&!std::is&#95;constructible&lt; T, &lt;a href="/thrust/api/classes/classoptional.html"&gt;optional&lt;/a&gt;&lt; U &gt; && &gt;::value &&!std::is&#95;constructible&lt; T, const &lt;a href="/thrust/api/classes/classoptional.html"&gt;optional&lt;/a&gt;&lt; U &gt; & &gt;::value &&!std::is&#95;constructible&lt; T, const &lt;a href="/thrust/api/classes/classoptional.html"&gt;optional&lt;/a&gt;&lt; U &gt; && &gt;::value &&!std::is&#95;convertible&lt; &lt;a href="/thrust/api/classes/classoptional.html"&gt;optional&lt;/a&gt;&lt; U &gt; &, T &gt;::value &&!std::is&#95;convertible&lt; &lt;a href="/thrust/api/classes/classoptional.html"&gt;optional&lt;/a&gt;&lt; U &gt; &&, T &gt;::value &&!std::is&#95;convertible&lt; const &lt;a href="/thrust/api/classes/classoptional.html"&gt;optional&lt;/a&gt;&lt; U &gt; &, T &gt;::value &&!std::is&#95;convertible&lt; const &lt;a href="/thrust/api/classes/classoptional.html"&gt;optional&lt;/a&gt;&lt; U &gt; &&, T &gt;::value &&!std::is&#95;assignable&lt; T &, &lt;a href="/thrust/api/classes/classoptional.html"&gt;optional&lt;/a&gt;&lt; U &gt; & &gt;::value &&!std::is&#95;assignable&lt; T &, &lt;a href="/thrust/api/classes/classoptional.html"&gt;optional&lt;/a&gt;&lt; U &gt; && &gt;::value &&!std::is&#95;assignable&lt; T &, const &lt;a href="/thrust/api/classes/classoptional.html"&gt;optional&lt;/a&gt;&lt; U &gt; & &gt;::value &&!std::is&#95;assignable&lt; T &, const &lt;a href="/thrust/api/classes/classoptional.html"&gt;optional&lt;/a&gt;&lt; U &gt; && &gt;::value &gt;;</span></code>

## Functions

<h3 id="function-allocator_delete_impl">
Function <code>detail::allocator&#95;delete&#95;impl</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Allocator,</span>
<span>&nbsp;&nbsp;typename Pointer&gt;</span>
<span>void </span><span><b>allocator_delete_impl</b>(Allocator const & alloc,</span>
<span>&nbsp;&nbsp;Pointer p,</span>
<span>&nbsp;&nbsp;std::false_type);</span></code>
<h3 id="function-allocator_delete_impl">
Function <code>detail::allocator&#95;delete&#95;impl</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Allocator,</span>
<span>&nbsp;&nbsp;typename Pointer&gt;</span>
<span>void </span><span><b>allocator_delete_impl</b>(Allocator const & alloc,</span>
<span>&nbsp;&nbsp;Pointer p,</span>
<span>&nbsp;&nbsp;std::true_type);</span></code>
<h3 id="function-array_allocator_delete_impl">
Function <code>detail::array&#95;allocator&#95;delete&#95;impl</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Allocator,</span>
<span>&nbsp;&nbsp;typename Pointer,</span>
<span>&nbsp;&nbsp;typename Size&gt;</span>
<span>void </span><span><b>array_allocator_delete_impl</b>(Allocator const & alloc,</span>
<span>&nbsp;&nbsp;Pointer p,</span>
<span>&nbsp;&nbsp;Size count,</span>
<span>&nbsp;&nbsp;std::false_type);</span></code>
<h3 id="function-array_allocator_delete_impl">
Function <code>detail::array&#95;allocator&#95;delete&#95;impl</code>
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
Function <code>detail::invoke</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Fn,</span>
<span>&nbsp;&nbsp;typename... Args,</span>
<span>&nbsp;&nbsp;typename = enable&#95;if&#95;t&lt;std::is&#95;member&#95;pointer&lt;decay&#95;t&lt;Fn&gt;&gt;::value&gt;,</span>
<span>&nbsp;&nbsp;int = 0&gt;</span>
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ auto </span><span><b>invoke</b>(Fn && f,</span>
<span>&nbsp;&nbsp;Args &&... args);</span></code>
<h3 id="function-invoke">
Function <code>detail::invoke</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Fn,</span>
<span>&nbsp;&nbsp;typename... Args,</span>
<span>&nbsp;&nbsp;typename = enable&#95;if&#95;t&lt;!std::is&#95;member&#95;pointer&lt;decay&#95;t&lt;Fn&gt;&gt;::value&gt;&gt;</span>
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ auto </span><span><b>invoke</b>(Fn && f,</span>
<span>&nbsp;&nbsp;Args &&... args);</span></code>
<h3 id="function-optional_map_impl">
Function <code>detail::optional&#95;map&#95;impl</code>
</h3>

<code class="doxybook">
<span>template &lt;class Opt,</span>
<span>&nbsp;&nbsp;class F,</span>
<span>&nbsp;&nbsp;class Ret = decltype(detail::invoke(std::declval&lt;F&gt;(),                                              &#42;std::declval&lt;Opt&gt;())),</span>
<span>&nbsp;&nbsp;detail::enable_if_t<!std::is_void< Ret >::value > * = nullptr&gt;</span>
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ auto </span><span><b>optional_map_impl</b>(Opt && opt,</span>
<span>&nbsp;&nbsp;F && f);</span></code>
<h3 id="function-optional_map_impl">
Function <code>detail::optional&#95;map&#95;impl</code>
</h3>

<code class="doxybook">
<span>template &lt;class Opt,</span>
<span>&nbsp;&nbsp;class F,</span>
<span>&nbsp;&nbsp;class Ret = decltype(detail::invoke(std::declval&lt;F&gt;(),                                              &#42;std::declval&lt;Opt&gt;())),</span>
<span>&nbsp;&nbsp;detail::enable_if_t< std::is_void< Ret >::value > * = nullptr&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ auto </span><span><b>optional_map_impl</b>(Opt && opt,</span>
<span>&nbsp;&nbsp;F && f);</span></code>

