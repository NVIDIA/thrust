---
title: optional
summary: An optional object is an object that contains the storage for another object and manages the lifetime of this contained object, if any. 
nav_exclude: true
has_children: true
has_toc: false
---

# Class `optional`

An optional object is an object that contains the storage for another object and manages the lifetime of this contained object, if any. 

The contained object may be initialized after the optional object has been initialized, and may be destroyed before the optional object has been destroyed. The initialization state of the contained object is tracked by the optional object. 

<code class="doxybook">
<span>#include <thrust/optional.h></span><br>
<span>template &lt;class T&gt;</span>
<span>class optional {</span>
<span>public:</span><span>&nbsp;&nbsp;using <b><a href="/thrust/api/classes/classoptional.html#using-value_type">value&#95;type</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group and_then Carries out some operation which returns an optional on the stored object if there is one.  */</span><span>&nbsp;&nbsp;template &lt;class F&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ auto </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-and_then">and&#95;then</a></b>(F && f);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group and_then \synopsis template <class F>\nconstexpr auto and_then(F &&f) &&;  */</span><span>&nbsp;&nbsp;template &lt;class F&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ auto </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-and_then">and&#95;then</a></b>(F && f);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group and_then \synopsis template <class F>\nconstexpr auto and_then(F &&f) const &;  */</span><span>&nbsp;&nbsp;template &lt;class F&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ auto </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-and_then">and&#95;then</a></b>(F && f) const;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group and_then \synopsis template <class F>\nconstexpr auto and_then(F &&f) const &&;  */</span><span>&nbsp;&nbsp;template &lt;class F&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ auto </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-and_then">and&#95;then</a></b>(F && f) const;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Carries out some operation on the stored object if there is one.  */</span><span>&nbsp;&nbsp;template &lt;class F&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ auto </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-map">map</a></b>(F && f);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group map \synopsis template <class F> constexpr auto map(F &&f) &&;  */</span><span>&nbsp;&nbsp;template &lt;class F&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ auto </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-map">map</a></b>(F && f);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group map \synopsis template <class F> constexpr auto map(F &&f) const&;  */</span><span>&nbsp;&nbsp;template &lt;class F&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ auto </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-map">map</a></b>(F && f) const;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group map \synopsis template <class F> constexpr auto map(F &&f) const&&;  */</span><span>&nbsp;&nbsp;template &lt;class F&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ auto </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-map">map</a></b>(F && f) const;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Calls <code>f</code> if the optional is empty \requires <code>std::invoke&#95;result&#95;t&lt;F&gt;</code> must be void or convertible to <code>optional&lt;T&gt;</code>.  */</span><span>&nbsp;&nbsp;template &lt;class F,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail::enable_if_ret_void< F > * = nullptr&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ constexpr <a href="/thrust/api/classes/classoptional.html">optional</a>< T > </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-or_else">or&#95;else</a></b>(F && f);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \exclude  */</span><span>&nbsp;&nbsp;template &lt;class F,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail::enable_if_ret_void< F > * = nullptr&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ constexpr <a href="/thrust/api/classes/classoptional.html">optional</a>< T > </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-or_else">or&#95;else</a></b>(F && f);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group or_else \synopsis template <class F> optional<T> or_else (F &&f) &&;  */</span><span>&nbsp;&nbsp;template &lt;class F,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail::enable_if_ret_void< F > * = nullptr&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ <a href="/thrust/api/classes/classoptional.html">optional</a>< T > </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-or_else">or&#95;else</a></b>(F && f);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \exclude  */</span><span>&nbsp;&nbsp;template &lt;class F,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail::disable_if_ret_void< F > * = nullptr&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ constexpr <a href="/thrust/api/classes/classoptional.html">optional</a>< T > </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-or_else">or&#95;else</a></b>(F && f);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group or_else \synopsis template <class F> optional<T> or_else (F &&f) const &;  */</span><span>&nbsp;&nbsp;template &lt;class F,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail::enable_if_ret_void< F > * = nullptr&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ <a href="/thrust/api/classes/classoptional.html">optional</a>< T > </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-or_else">or&#95;else</a></b>(F && f) const;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \exclude  */</span><span>&nbsp;&nbsp;template &lt;class F,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail::disable_if_ret_void< F > * = nullptr&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ constexpr <a href="/thrust/api/classes/classoptional.html">optional</a>< T > </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-or_else">or&#95;else</a></b>(F && f) const;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \exclude  */</span><span>&nbsp;&nbsp;template &lt;class F,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail::enable_if_ret_void< F > * = nullptr&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ <a href="/thrust/api/classes/classoptional.html">optional</a>< T > </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-or_else">or&#95;else</a></b>(F && f) const;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \exclude  */</span><span>&nbsp;&nbsp;template &lt;class F,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail::enable_if_ret_void< F > * = nullptr&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ <a href="/thrust/api/classes/classoptional.html">optional</a>< T > </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-or_else">or&#95;else</a></b>(F && f) const;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Maps the stored value with <code>f</code> if there is one, otherwise returns <code>u</code>.  */</span><span>&nbsp;&nbsp;template &lt;class F,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;class U&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ U </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-map_or">map&#95;or</a></b>(F && f,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;U && u);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group map_or  */</span><span>&nbsp;&nbsp;template &lt;class F,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;class U&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ U </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-map_or">map&#95;or</a></b>(F && f,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;U && u);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group map_or  */</span><span>&nbsp;&nbsp;template &lt;class F,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;class U&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ U </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-map_or">map&#95;or</a></b>(F && f,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;U && u) const;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group map_or  */</span><span>&nbsp;&nbsp;template &lt;class F,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;class U&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ U </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-map_or">map&#95;or</a></b>(F && f,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;U && u) const;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Maps the stored value with <code>f</code> if there is one, otherwise calls <code>u</code> and returns the result.  */</span><span>&nbsp;&nbsp;template &lt;class F,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;class U&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ detail::invoke_result_t< U > </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-map_or_else">map&#95;or&#95;else</a></b>(F && f,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;U && u);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group map_or_else \synopsis template <class F, class U>\nauto map_or_else(F &&f, U &&u) &&;  */</span><span>&nbsp;&nbsp;template &lt;class F,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;class U&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ detail::invoke_result_t< U > </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-map_or_else">map&#95;or&#95;else</a></b>(F && f,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;U && u);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group map_or_else \synopsis template <class F, class U>\nauto map_or_else(F &&f, U &&u) const &;  */</span><span>&nbsp;&nbsp;template &lt;class F,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;class U&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ detail::invoke_result_t< U > </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-map_or_else">map&#95;or&#95;else</a></b>(F && f,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;U && u) const;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group map_or_else \synopsis template <class F, class U>\nauto map_or_else(F &&f, U &&u) const &&;  */</span><span>&nbsp;&nbsp;template &lt;class F,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;class U&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ detail::invoke_result_t< U > </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-map_or_else">map&#95;or&#95;else</a></b>(F && f,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;U && u) const;</span>
<br>
<span>&nbsp;&nbsp;template &lt;class U&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ <a href="/thrust/api/classes/classoptional.html">optional</a>< typename std::decay< U >::type > </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-conjunction">conjunction</a></b>(U && u) const;</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-disjunction">disjunction</a></b>(const <a href="/thrust/api/classes/classoptional.html">optional</a> & rhs);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group disjunction  */</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-disjunction">disjunction</a></b>(const <a href="/thrust/api/classes/classoptional.html">optional</a> & rhs) const;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group disjunction  */</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-disjunction">disjunction</a></b>(const <a href="/thrust/api/classes/classoptional.html">optional</a> & rhs);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group disjunction  */</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-disjunction">disjunction</a></b>(const <a href="/thrust/api/classes/classoptional.html">optional</a> & rhs) const;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group disjunction  */</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-disjunction">disjunction</a></b>(<a href="/thrust/api/classes/classoptional.html">optional</a> && rhs);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group disjunction  */</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-disjunction">disjunction</a></b>(<a href="/thrust/api/classes/classoptional.html">optional</a> && rhs) const;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group disjunction  */</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-disjunction">disjunction</a></b>(<a href="/thrust/api/classes/classoptional.html">optional</a> && rhs);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group disjunction  */</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-disjunction">disjunction</a></b>(<a href="/thrust/api/classes/classoptional.html">optional</a> && rhs) const;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Takes the value out of the optional, leaving it empty \group take.  */</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-take">take</a></b>();</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group take  */</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-take">take</a></b>() const;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group take  */</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-take">take</a></b>();</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group take  */</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-take">take</a></b>() const;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Constructs an optional that does not contain a value.  */</span><span>&nbsp;&nbsp;constexpr __thrust_exec_check_disable__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-optional">optional</a></b>() = default;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group ctor_empty  */</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-optional">optional</a></b>(<a href="/thrust/api/classes/structnullopt__t.html">nullopt_t</a>);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Copy constructor.  */</span><span>&nbsp;&nbsp;constexpr __thrust_exec_check_disable__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-optional">optional</a></b>(const <a href="/thrust/api/classes/classoptional.html">optional</a> & rhs) = default;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Move constructor.  */</span><span>&nbsp;&nbsp;constexpr __thrust_exec_check_disable__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-optional">optional</a></b>(<a href="/thrust/api/classes/classoptional.html">optional</a> && rhs) = default;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Constructs the stored value in-place using the given arguments.  */</span><span>&nbsp;&nbsp;template &lt;class... Args&gt;</span>
<span>&nbsp;&nbsp;explicit __thrust_exec_check_disable__ __host__ constexpr __device__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-optional">optional</a></b>(detail::enable_if_t< std::is_constructible< T, Args... >::<a href="/thrust/api/classes/classoptional.html#function-value">value</a>, <a href="/thrust/api/classes/structin__place__t.html">in_place_t</a> >,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;Args &&... args);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group in_place \synopsis template <class U, class...  */</span><span>&nbsp;&nbsp;template &lt;class U,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;class... Args&gt;</span>
<span>&nbsp;&nbsp;explicit __thrust_exec_check_disable__ __host__ constexpr __device__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-optional">optional</a></b>(detail::enable_if_t< std::is_constructible< T, std::initializer_list< U > &, Args &&... >::<a href="/thrust/api/classes/classoptional.html#function-value">value</a>, <a href="/thrust/api/classes/structin__place__t.html">in_place_t</a> >,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;std::initializer_list< U > il,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;Args &&... args);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Constructs the stored value with <code>u</code>.  */</span><span>&nbsp;&nbsp;template &lt;class U = T,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail::enable_if_t< std::is_convertible< U &&, T >::<a href="/thrust/api/classes/classoptional.html#function-value">value</a> > * = nullptr,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail::enable_forward_value< T, U > * = nullptr&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-optional">optional</a></b>(U && u);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \exclude  */</span><span>&nbsp;&nbsp;template &lt;class U = T,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail::enable_if_t< std::is_convertible< U &&, T >::<a href="/thrust/api/classes/classoptional.html#function-value">value</a> > * = nullptr,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail::enable_forward_value< T, U > * = nullptr&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-optional">optional</a></b>(U && u);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Converting copy constructor.  */</span><span>&nbsp;&nbsp;template &lt;class U,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail::enable_from_other< T, U, const U & > * = nullptr,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail::enable_if_t< std::is_convertible< const U &, T >::<a href="/thrust/api/classes/classoptional.html#function-value">value</a> > * = nullptr&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-optional">optional</a></b>(const <a href="/thrust/api/classes/classoptional.html">optional</a>< U > & rhs);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \exclude  */</span><span>&nbsp;&nbsp;template &lt;class U,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail::enable_from_other< T, U, const U & > * = nullptr,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail::enable_if_t< std::is_convertible< const U &, T >::<a href="/thrust/api/classes/classoptional.html#function-value">value</a> > * = nullptr&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-optional">optional</a></b>(const <a href="/thrust/api/classes/classoptional.html">optional</a>< U > & rhs);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Converting move constructor.  */</span><span>&nbsp;&nbsp;template &lt;class U,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail::enable_from_other< T, U, U && > * = nullptr,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail::enable_if_t< std::is_convertible< U &&, T >::<a href="/thrust/api/classes/classoptional.html#function-value">value</a> > * = nullptr&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-optional">optional</a></b>(<a href="/thrust/api/classes/classoptional.html">optional</a>< U > && rhs);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \exclude  */</span><span>&nbsp;&nbsp;template &lt;class U,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail::enable_from_other< T, U, U && > * = nullptr,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail::enable_if_t< std::is_convertible< U &&, T >::<a href="/thrust/api/classes/classoptional.html#function-value">value</a> > * = nullptr&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-optional">optional</a></b>(<a href="/thrust/api/classes/classoptional.html">optional</a>< U > && rhs);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Destroys the stored value if there is one.  */</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-~optional">~optional</a></b>() = default;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Assignment to empty.  */</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> & </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-operator=">operator=</a></b>(<a href="/thrust/api/classes/structnullopt__t.html">nullopt_t</a>);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Copy assignment.  */</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ <a href="/thrust/api/classes/classoptional.html">optional</a> & </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-operator=">operator=</a></b>(const <a href="/thrust/api/classes/classoptional.html">optional</a> & rhs) = default;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Move assignment.  */</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ <a href="/thrust/api/classes/classoptional.html">optional</a> & </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-operator=">operator=</a></b>(<a href="/thrust/api/classes/classoptional.html">optional</a> && rhs) = default;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Assigns the stored value from <code>u</code>, destroying the old value if there was one.  */</span><span>&nbsp;&nbsp;template &lt;class U = T,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail::enable_assign_forward< T, U > * = nullptr&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> & </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-operator=">operator=</a></b>(U && u);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Converting copy assignment operator.  */</span><span>&nbsp;&nbsp;template &lt;class U,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail::enable_assign_from_other< T, U, const U & > * = nullptr&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> & </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-operator=">operator=</a></b>(const <a href="/thrust/api/classes/classoptional.html">optional</a>< U > & rhs);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Converting move assignment operator.  */</span><span>&nbsp;&nbsp;template &lt;class U,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail::enable_assign_from_other< T, U, U > * = nullptr&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> & </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-operator=">operator=</a></b>(<a href="/thrust/api/classes/classoptional.html">optional</a>< U > && rhs);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Constructs the value in-place, destroying the current one if there is one.  */</span><span>&nbsp;&nbsp;template &lt;class... Args&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ T & </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-emplace">emplace</a></b>(Args &&... args);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group emplace \synopsis template <class U, class...  */</span><span>&nbsp;&nbsp;template &lt;class U,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;class... Args&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ detail::enable_if_t< std::is_constructible< T, std::initializer_list< U > &, Args &&... >::<a href="/thrust/api/classes/classoptional.html#function-value">value</a>, T & > </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-emplace">emplace</a></b>(std::initializer_list< U > il,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;Args &&... args);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Swaps this optional with the other.  */</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ void </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-swap">swap</a></b>(<a href="/thrust/api/classes/classoptional.html">optional</a> & rhs);</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr const __device__ T * </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-operator->">operator-&gt;</a></b>() const;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group pointer \synopsis constexpr T *operator->();  */</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ T * </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-operator->">operator-&gt;</a></b>();</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ T & </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-operator*">operator&#42;</a></b>();</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group deref \synopsis constexpr const T &operator*() const;  */</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr const __device__ T & </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-operator*">operator&#42;</a></b>() const;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \exclude  */</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ T && </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-operator*">operator&#42;</a></b>();</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \exclude  */</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr const __device__ T && </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-operator*">operator&#42;</a></b>() const;</span>
<br>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ bool </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-has_value">has&#95;value</a></b>() const;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group has_value  */</span><span>&nbsp;&nbsp;explicit __thrust_exec_check_disable__ __host__ constexpr __device__ </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-operator-bool">operator bool</a></b>() const;</span>
<br>
<span>&nbsp;&nbsp;constexpr __host__ T & </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-value">value</a></b>();</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group value \synopsis constexpr const T &value() const;  */</span><span>&nbsp;&nbsp;constexpr const __host__ T & </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-value">value</a></b>() const;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \exclude  */</span><span>&nbsp;&nbsp;constexpr __host__ T && </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-value">value</a></b>();</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \exclude  */</span><span>&nbsp;&nbsp;constexpr const __host__ T && </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-value">value</a></b>() const;</span>
<br>
<span>&nbsp;&nbsp;template &lt;class U&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ T </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-value_or">value&#95;or</a></b>(U && u) const;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* \group value_or  */</span><span>&nbsp;&nbsp;template &lt;class U&gt;</span>
<span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ constexpr __device__ T </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-value_or">value&#95;or</a></b>(U && u);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Destroys the stored value if one exists, making the optional empty.  */</span><span>&nbsp;&nbsp;__thrust_exec_check_disable__ __host__ __device__ void </span><span>&nbsp;&nbsp;<b><a href="/thrust/api/classes/classoptional.html#function-reset">reset</a></b>();</span>
<span>};</span>
</code>

## Member Types

<h3 id="using-value_type">
Type Alias <code>optional::value&#95;type</code>
</h3>

<code class="doxybook">
<span>using <b>value_type</b> = T;</span></code>

## Member Functions

<h3 id="function-and_then">
Function <code>optional::&gt;::and&#95;then</code>
</h3>

<code class="doxybook">
<span>template &lt;class F&gt;</span>
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ auto </span><span><b>and_then</b>(F && f);</span></code>
\group and_then Carries out some operation which returns an optional on the stored object if there is one. 

\requires <code>std::invoke(std::forward&lt;F&gt;(f), <a href="/thrust/api/classes/classoptional.html#function-value">value()</a>)</code> returns a <code>std::optional&lt;U&gt;</code> for some <code>U</code>. 

**Returns**:
Let <code>U</code> be the result of <code>std::invoke(std::forward&lt;F&gt;(f), <a href="/thrust/api/classes/classoptional.html#function-value">value()</a>)</code>. Returns a <code>std::optional&lt;U&gt;</code>. The return value is empty if <code>&#42;this</code> is empty, otherwise the return value of <code>std::invoke(std::forward&lt;F&gt;(f), <a href="/thrust/api/classes/classoptional.html#function-value">value()</a>)</code> is returned. \group and_then \synopsis template <class F>\nconstexpr auto and_then(F &&f) &; 

<h3 id="function-and_then">
Function <code>optional::&gt;::and&#95;then</code>
</h3>

<code class="doxybook">
<span>template &lt;class F&gt;</span>
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ auto </span><span><b>and_then</b>(F && f);</span></code>
\group and_then \synopsis template <class F>\nconstexpr auto and_then(F &&f) &&; 

<h3 id="function-and_then">
Function <code>optional::&gt;::and&#95;then</code>
</h3>

<code class="doxybook">
<span>template &lt;class F&gt;</span>
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ auto </span><span><b>and_then</b>(F && f) const;</span></code>
\group and_then \synopsis template <class F>\nconstexpr auto and_then(F &&f) const &; 

<h3 id="function-and_then">
Function <code>optional::&gt;::and&#95;then</code>
</h3>

<code class="doxybook">
<span>template &lt;class F&gt;</span>
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ auto </span><span><b>and_then</b>(F && f) const;</span></code>
\group and_then \synopsis template <class F>\nconstexpr auto and_then(F &&f) const &&; 

<h3 id="function-map">
Function <code>optional::&gt;::map</code>
</h3>

<code class="doxybook">
<span>template &lt;class F&gt;</span>
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ auto </span><span><b>map</b>(F && f);</span></code>
Carries out some operation on the stored object if there is one. 


\group map \synopsis template <class F> constexpr auto map(F &&f) &; 

**Returns**:
Let <code>U</code> be the result of <code>std::invoke(std::forward&lt;F&gt;(f), <a href="/thrust/api/classes/classoptional.html#function-value">value()</a>)</code>. Returns a <code>std::optional&lt;U&gt;</code>. The return value is empty if <code>&#42;this</code> is empty, otherwise an <code>optional&lt;U&gt;</code> is constructed from the return value of <code>std::invoke(std::forward&lt;F&gt;(f), <a href="/thrust/api/classes/classoptional.html#function-value">value()</a>)</code> and is returned.

<h3 id="function-map">
Function <code>optional::&gt;::map</code>
</h3>

<code class="doxybook">
<span>template &lt;class F&gt;</span>
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ auto </span><span><b>map</b>(F && f);</span></code>
\group map \synopsis template <class F> constexpr auto map(F &&f) &&; 

<h3 id="function-map">
Function <code>optional::&gt;::map</code>
</h3>

<code class="doxybook">
<span>template &lt;class F&gt;</span>
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ auto </span><span><b>map</b>(F && f) const;</span></code>
\group map \synopsis template <class F> constexpr auto map(F &&f) const&; 

<h3 id="function-map">
Function <code>optional::&gt;::map</code>
</h3>

<code class="doxybook">
<span>template &lt;class F&gt;</span>
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ auto </span><span><b>map</b>(F && f) const;</span></code>
\group map \synopsis template <class F> constexpr auto map(F &&f) const&&; 

<h3 id="function-or_else">
Function <code>optional::&gt;::or&#95;else</code>
</h3>

<code class="doxybook">
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;detail::enable_if_ret_void< F > * = nullptr&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ constexpr <a href="/thrust/api/classes/classoptional.html">optional</a>< T > </span><span><b>or_else</b>(F && f);</span></code>
Calls <code>f</code> if the optional is empty \requires <code>std::invoke&#95;result&#95;t&lt;F&gt;</code> must be void or convertible to <code>optional&lt;T&gt;</code>. 

\effects If <code>&#42;this</code> has a value, returns <code>&#42;this</code>. Otherwise, if <code>f</code> returns <code>void</code>, calls <code>std::forward&lt;F&gt;(f)</code> and returns <code>std::nullopt</code>. Otherwise, returns <code>std::forward&lt;F&gt;(f)()</code>.

\group or_else \synopsis template <class F> optional<T> or_else (F &&f) &; 

<h3 id="function-or_else">
Function <code>optional::&gt;::or&#95;else</code>
</h3>

<code class="doxybook">
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;detail::enable_if_ret_void< F > * = nullptr&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ constexpr <a href="/thrust/api/classes/classoptional.html">optional</a>< T > </span><span><b>or_else</b>(F && f);</span></code>
\exclude 

\effects If <code>&#42;this</code> has a value, returns <code>&#42;this</code>. Otherwise, if <code>f</code> returns <code>void</code>, calls <code>std::forward&lt;F&gt;(f)</code> and returns <code>std::nullopt</code>. Otherwise, returns <code>std::forward&lt;F&gt;(f)()</code>.

\group or_else \synopsis template <class F> optional<T> or_else (F &&f) &; 

<h3 id="function-or_else">
Function <code>optional::&gt;::or&#95;else</code>
</h3>

<code class="doxybook">
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;detail::enable_if_ret_void< F > * = nullptr&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ <a href="/thrust/api/classes/classoptional.html">optional</a>< T > </span><span><b>or_else</b>(F && f);</span></code>
\group or_else \synopsis template <class F> optional<T> or_else (F &&f) &&; 

<h3 id="function-or_else">
Function <code>optional::&gt;::or&#95;else</code>
</h3>

<code class="doxybook">
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;detail::disable_if_ret_void< F > * = nullptr&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ constexpr <a href="/thrust/api/classes/classoptional.html">optional</a>< T > </span><span><b>or_else</b>(F && f);</span></code>
\exclude 

<h3 id="function-or_else">
Function <code>optional::&gt;::or&#95;else</code>
</h3>

<code class="doxybook">
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;detail::enable_if_ret_void< F > * = nullptr&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ <a href="/thrust/api/classes/classoptional.html">optional</a>< T > </span><span><b>or_else</b>(F && f) const;</span></code>
\group or_else \synopsis template <class F> optional<T> or_else (F &&f) const &; 

<h3 id="function-or_else">
Function <code>optional::&gt;::or&#95;else</code>
</h3>

<code class="doxybook">
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;detail::disable_if_ret_void< F > * = nullptr&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ constexpr <a href="/thrust/api/classes/classoptional.html">optional</a>< T > </span><span><b>or_else</b>(F && f) const;</span></code>
\exclude 

<h3 id="function-or_else">
Function <code>optional::&gt;::or&#95;else</code>
</h3>

<code class="doxybook">
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;detail::enable_if_ret_void< F > * = nullptr&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ <a href="/thrust/api/classes/classoptional.html">optional</a>< T > </span><span><b>or_else</b>(F && f) const;</span></code>
\exclude 

<h3 id="function-or_else">
Function <code>optional::&gt;::or&#95;else</code>
</h3>

<code class="doxybook">
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;detail::enable_if_ret_void< F > * = nullptr&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ <a href="/thrust/api/classes/classoptional.html">optional</a>< T > </span><span><b>or_else</b>(F && f) const;</span></code>
\exclude 

<h3 id="function-map_or">
Function <code>optional::&gt;::map&#95;or</code>
</h3>

<code class="doxybook">
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class U&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ U </span><span><b>map_or</b>(F && f,</span>
<span>&nbsp;&nbsp;U && u);</span></code>
Maps the stored value with <code>f</code> if there is one, otherwise returns <code>u</code>. 

If there is a value stored, then <code>f</code> is called with <code>&#42;&#42;this</code> and the value is returned. Otherwise <code>u</code> is returned.

\group map_or 

<h3 id="function-map_or">
Function <code>optional::&gt;::map&#95;or</code>
</h3>

<code class="doxybook">
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class U&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ U </span><span><b>map_or</b>(F && f,</span>
<span>&nbsp;&nbsp;U && u);</span></code>
\group map_or 

<h3 id="function-map_or">
Function <code>optional::&gt;::map&#95;or</code>
</h3>

<code class="doxybook">
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class U&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ U </span><span><b>map_or</b>(F && f,</span>
<span>&nbsp;&nbsp;U && u) const;</span></code>
\group map_or 

<h3 id="function-map_or">
Function <code>optional::&gt;::map&#95;or</code>
</h3>

<code class="doxybook">
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class U&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ U </span><span><b>map_or</b>(F && f,</span>
<span>&nbsp;&nbsp;U && u) const;</span></code>
\group map_or 

<h3 id="function-map_or_else">
Function <code>optional::&gt;::map&#95;or&#95;else</code>
</h3>

<code class="doxybook">
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class U&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ detail::invoke_result_t< U > </span><span><b>map_or_else</b>(F && f,</span>
<span>&nbsp;&nbsp;U && u);</span></code>
Maps the stored value with <code>f</code> if there is one, otherwise calls <code>u</code> and returns the result. 

If there is a value stored, then <code>f</code> is called with <code>&#42;&#42;this</code> and the value is returned. Otherwise <code>std::forward&lt;U&gt;(u)()</code> is returned.

\group map_or_else \synopsis template <class F, class U>\nauto map_or_else(F &&f, U &&u) &; 

<h3 id="function-map_or_else">
Function <code>optional::&gt;::map&#95;or&#95;else</code>
</h3>

<code class="doxybook">
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class U&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ detail::invoke_result_t< U > </span><span><b>map_or_else</b>(F && f,</span>
<span>&nbsp;&nbsp;U && u);</span></code>
\group map_or_else \synopsis template <class F, class U>\nauto map_or_else(F &&f, U &&u) &&; 

<h3 id="function-map_or_else">
Function <code>optional::&gt;::map&#95;or&#95;else</code>
</h3>

<code class="doxybook">
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class U&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ detail::invoke_result_t< U > </span><span><b>map_or_else</b>(F && f,</span>
<span>&nbsp;&nbsp;U && u) const;</span></code>
\group map_or_else \synopsis template <class F, class U>\nauto map_or_else(F &&f, U &&u) const &; 

<h3 id="function-map_or_else">
Function <code>optional::&gt;::map&#95;or&#95;else</code>
</h3>

<code class="doxybook">
<span>template &lt;class F,</span>
<span>&nbsp;&nbsp;class U&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ detail::invoke_result_t< U > </span><span><b>map_or_else</b>(F && f,</span>
<span>&nbsp;&nbsp;U && u) const;</span></code>
\group map_or_else \synopsis template <class F, class U>\nauto map_or_else(F &&f, U &&u) const &&; 

<h3 id="function-conjunction">
Function <code>optional::&gt;::conjunction</code>
</h3>

<code class="doxybook">
<span>template &lt;class U&gt;</span>
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ <a href="/thrust/api/classes/classoptional.html">optional</a>< typename std::decay< U >::type > </span><span><b>conjunction</b>(U && u) const;</span></code>
**Returns**:
<code>u</code> if <code>&#42;this</code> has a value, otherwise an empty optional. 

<h3 id="function-disjunction">
Function <code>optional::&gt;::disjunction</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> </span><span><b>disjunction</b>(const <a href="/thrust/api/classes/classoptional.html">optional</a> & rhs);</span></code>
**Returns**:
<code>rhs</code> if <code>&#42;this</code> is empty, otherwise the current value. \group disjunction 

<h3 id="function-disjunction">
Function <code>optional::&gt;::disjunction</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> </span><span><b>disjunction</b>(const <a href="/thrust/api/classes/classoptional.html">optional</a> & rhs) const;</span></code>
\group disjunction 

<h3 id="function-disjunction">
Function <code>optional::&gt;::disjunction</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> </span><span><b>disjunction</b>(const <a href="/thrust/api/classes/classoptional.html">optional</a> & rhs);</span></code>
\group disjunction 

<h3 id="function-disjunction">
Function <code>optional::&gt;::disjunction</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> </span><span><b>disjunction</b>(const <a href="/thrust/api/classes/classoptional.html">optional</a> & rhs) const;</span></code>
\group disjunction 

<h3 id="function-disjunction">
Function <code>optional::&gt;::disjunction</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> </span><span><b>disjunction</b>(<a href="/thrust/api/classes/classoptional.html">optional</a> && rhs);</span></code>
\group disjunction 

<h3 id="function-disjunction">
Function <code>optional::&gt;::disjunction</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> </span><span><b>disjunction</b>(<a href="/thrust/api/classes/classoptional.html">optional</a> && rhs) const;</span></code>
\group disjunction 

<h3 id="function-disjunction">
Function <code>optional::&gt;::disjunction</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> </span><span><b>disjunction</b>(<a href="/thrust/api/classes/classoptional.html">optional</a> && rhs);</span></code>
\group disjunction 

<h3 id="function-disjunction">
Function <code>optional::&gt;::disjunction</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> </span><span><b>disjunction</b>(<a href="/thrust/api/classes/classoptional.html">optional</a> && rhs) const;</span></code>
\group disjunction 

<h3 id="function-take">
Function <code>optional::&gt;::take</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> </span><span><b>take</b>();</span></code>
Takes the value out of the optional, leaving it empty \group take. 

<h3 id="function-take">
Function <code>optional::&gt;::take</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> </span><span><b>take</b>() const;</span></code>
\group take 

<h3 id="function-take">
Function <code>optional::&gt;::take</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> </span><span><b>take</b>();</span></code>
\group take 

<h3 id="function-take">
Function <code>optional::&gt;::take</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> </span><span><b>take</b>() const;</span></code>
\group take 

<h3 id="function-optional">
Function <code>optional::&gt;::optional</code>
</h3>

<code class="doxybook">
<span>constexpr __thrust_exec_check_disable__ </span><span><b>optional</b>() = default;</span></code>
Constructs an optional that does not contain a value. 

\group ctor_empty 

<h3 id="function-optional">
Function <code>optional::&gt;::optional</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ </span><span><b>optional</b>(<a href="/thrust/api/classes/structnullopt__t.html">nullopt_t</a>);</span></code>
\group ctor_empty 

<h3 id="function-optional">
Function <code>optional::&gt;::optional</code>
</h3>

<code class="doxybook">
<span>constexpr __thrust_exec_check_disable__ </span><span><b>optional</b>(const <a href="/thrust/api/classes/classoptional.html">optional</a> & rhs) = default;</span></code>
Copy constructor. 

If <code>rhs</code> contains a value, the stored value is direct-initialized with it. Otherwise, the constructed optional is empty. 

<h3 id="function-optional">
Function <code>optional::&gt;::optional</code>
</h3>

<code class="doxybook">
<span>constexpr __thrust_exec_check_disable__ </span><span><b>optional</b>(<a href="/thrust/api/classes/classoptional.html">optional</a> && rhs) = default;</span></code>
Move constructor. 

If <code>rhs</code> contains a value, the stored value is direct-initialized with it. Otherwise, the constructed optional is empty. 

<h3 id="function-optional">
Function <code>optional::&gt;::optional</code>
</h3>

<code class="doxybook">
<span>template &lt;class... Args&gt;</span>
<span>explicit __thrust_exec_check_disable__ __host__ constexpr __device__ </span><span><b>optional</b>(detail::enable_if_t< std::is_constructible< T, Args... >::<a href="/thrust/api/classes/classoptional.html#function-value">value</a>, <a href="/thrust/api/classes/structin__place__t.html">in_place_t</a> >,</span>
<span>&nbsp;&nbsp;Args &&... args);</span></code>
Constructs the stored value in-place using the given arguments. 

\group in_place \synopsis template <class... Args> constexpr explicit <a href="/thrust/api/classes/classoptional.html">optional(in_place_t, Args&&... args)</a>; 

<h3 id="function-optional">
Function <code>optional::&gt;::optional</code>
</h3>

<code class="doxybook">
<span>template &lt;class U,</span>
<span>&nbsp;&nbsp;class... Args&gt;</span>
<span>explicit __thrust_exec_check_disable__ __host__ constexpr __device__ </span><span><b>optional</b>(detail::enable_if_t< std::is_constructible< T, std::initializer_list< U > &, Args &&... >::<a href="/thrust/api/classes/classoptional.html#function-value">value</a>, <a href="/thrust/api/classes/structin__place__t.html">in_place_t</a> >,</span>
<span>&nbsp;&nbsp;std::initializer_list< U > il,</span>
<span>&nbsp;&nbsp;Args &&... args);</span></code>
\group in_place \synopsis template <class U, class... 

Args>\nconstexpr explicit <a href="/thrust/api/classes/classoptional.html">optional(in_place_t, std::initializer_list<U>&, Args&&... args)</a>; 

<h3 id="function-optional">
Function <code>optional::&gt;::optional</code>
</h3>

<code class="doxybook">
<span>template &lt;class U = T,</span>
<span>&nbsp;&nbsp;detail::enable_if_t< std::is_convertible< U &&, T >::<a href="/thrust/api/classes/classoptional.html#function-value">value</a> > * = nullptr,</span>
<span>&nbsp;&nbsp;detail::enable_forward_value< T, U > * = nullptr&gt;</span>
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ </span><span><b>optional</b>(U && u);</span></code>
Constructs the stored value with <code>u</code>. 

\synopsis template <class U=T>> constexpr <a href="/thrust/api/classes/classoptional.html#function-optional">optional(U &&u)</a>; 

<h3 id="function-optional">
Function <code>optional::&gt;::optional</code>
</h3>

<code class="doxybook">
<span>template &lt;class U = T,</span>
<span>&nbsp;&nbsp;detail::enable_if_t< std::is_convertible< U &&, T >::<a href="/thrust/api/classes/classoptional.html#function-value">value</a> > * = nullptr,</span>
<span>&nbsp;&nbsp;detail::enable_forward_value< T, U > * = nullptr&gt;</span>
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ </span><span><b>optional</b>(U && u);</span></code>
\exclude 

\synopsis template <class U=T>> constexpr <a href="/thrust/api/classes/classoptional.html#function-optional">optional(U &&u)</a>; 

<h3 id="function-optional">
Function <code>optional::&gt;::optional</code>
</h3>

<code class="doxybook">
<span>template &lt;class U,</span>
<span>&nbsp;&nbsp;detail::enable_from_other< T, U, const U & > * = nullptr,</span>
<span>&nbsp;&nbsp;detail::enable_if_t< std::is_convertible< const U &, T >::<a href="/thrust/api/classes/classoptional.html#function-value">value</a> > * = nullptr&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ </span><span><b>optional</b>(const <a href="/thrust/api/classes/classoptional.html">optional</a>< U > & rhs);</span></code>
Converting copy constructor. 

\synopsis template <class U> <a href="/thrust/api/classes/classoptional.html#function-optional">optional(const optional<U> &rhs)</a>; 

<h3 id="function-optional">
Function <code>optional::&gt;::optional</code>
</h3>

<code class="doxybook">
<span>template &lt;class U,</span>
<span>&nbsp;&nbsp;detail::enable_from_other< T, U, const U & > * = nullptr,</span>
<span>&nbsp;&nbsp;detail::enable_if_t< std::is_convertible< const U &, T >::<a href="/thrust/api/classes/classoptional.html#function-value">value</a> > * = nullptr&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ </span><span><b>optional</b>(const <a href="/thrust/api/classes/classoptional.html">optional</a>< U > & rhs);</span></code>
\exclude 

\synopsis template <class U> <a href="/thrust/api/classes/classoptional.html#function-optional">optional(const optional<U> &rhs)</a>; 

<h3 id="function-optional">
Function <code>optional::&gt;::optional</code>
</h3>

<code class="doxybook">
<span>template &lt;class U,</span>
<span>&nbsp;&nbsp;detail::enable_from_other< T, U, U && > * = nullptr,</span>
<span>&nbsp;&nbsp;detail::enable_if_t< std::is_convertible< U &&, T >::<a href="/thrust/api/classes/classoptional.html#function-value">value</a> > * = nullptr&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ </span><span><b>optional</b>(<a href="/thrust/api/classes/classoptional.html">optional</a>< U > && rhs);</span></code>
Converting move constructor. 

\synopsis template <class U> <a href="/thrust/api/classes/classoptional.html#function-optional">optional(optional<U> &&rhs)</a>; 

<h3 id="function-optional">
Function <code>optional::&gt;::optional</code>
</h3>

<code class="doxybook">
<span>template &lt;class U,</span>
<span>&nbsp;&nbsp;detail::enable_from_other< T, U, U && > * = nullptr,</span>
<span>&nbsp;&nbsp;detail::enable_if_t< std::is_convertible< U &&, T >::<a href="/thrust/api/classes/classoptional.html#function-value">value</a> > * = nullptr&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ </span><span><b>optional</b>(<a href="/thrust/api/classes/classoptional.html">optional</a>< U > && rhs);</span></code>
\exclude 

\synopsis template <class U> <a href="/thrust/api/classes/classoptional.html#function-optional">optional(optional<U> &&rhs)</a>; 

<h3 id="function-~optional">
Function <code>optional::&gt;::~optional</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ </span><span><b>~optional</b>() = default;</span></code>
Destroys the stored value if there is one. 

<h3 id="function-operator=">
Function <code>optional::&gt;::operator=</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> & </span><span><b>operator=</b>(<a href="/thrust/api/classes/structnullopt__t.html">nullopt_t</a>);</span></code>
Assignment to empty. 

Destroys the current value if there is one. 

<h3 id="function-operator=">
Function <code>optional::&gt;::operator=</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ <a href="/thrust/api/classes/classoptional.html">optional</a> & </span><span><b>operator=</b>(const <a href="/thrust/api/classes/classoptional.html">optional</a> & rhs) = default;</span></code>
Copy assignment. 

Copies the value from <code>rhs</code> if there is one. Otherwise resets the stored value in <code>&#42;this</code>. 

<h3 id="function-operator=">
Function <code>optional::&gt;::operator=</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ <a href="/thrust/api/classes/classoptional.html">optional</a> & </span><span><b>operator=</b>(<a href="/thrust/api/classes/classoptional.html">optional</a> && rhs) = default;</span></code>
Move assignment. 

Moves the value from <code>rhs</code> if there is one. Otherwise resets the stored value in <code>&#42;this</code>. 

<h3 id="function-operator=">
Function <code>optional::&gt;::operator=</code>
</h3>

<code class="doxybook">
<span>template &lt;class U = T,</span>
<span>&nbsp;&nbsp;detail::enable_assign_forward< T, U > * = nullptr&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> & </span><span><b>operator=</b>(U && u);</span></code>
Assigns the stored value from <code>u</code>, destroying the old value if there was one. 

\synopsis optional &<a href="/thrust/api/classes/classoptional.html#function-operator=">operator=(U &&u)</a>; 

<h3 id="function-operator=">
Function <code>optional::&gt;::operator=</code>
</h3>

<code class="doxybook">
<span>template &lt;class U,</span>
<span>&nbsp;&nbsp;detail::enable_assign_from_other< T, U, const U & > * = nullptr&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> & </span><span><b>operator=</b>(const <a href="/thrust/api/classes/classoptional.html">optional</a>< U > & rhs);</span></code>
Converting copy assignment operator. 

Copies the value from <code>rhs</code> if there is one. Otherwise resets the stored value in <code>&#42;this</code>. \synopsis optional &<a href="/thrust/api/classes/classoptional.html#function-operator=">operator=(const optional<U> & rhs)</a>; 

<h3 id="function-operator=">
Function <code>optional::&gt;::operator=</code>
</h3>

<code class="doxybook">
<span>template &lt;class U,</span>
<span>&nbsp;&nbsp;detail::enable_assign_from_other< T, U, U > * = nullptr&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ <a href="/thrust/api/classes/classoptional.html">optional</a> & </span><span><b>operator=</b>(<a href="/thrust/api/classes/classoptional.html">optional</a>< U > && rhs);</span></code>
Converting move assignment operator. 

Moves the value from <code>rhs</code> if there is one. Otherwise resets the stored value in <code>&#42;this</code>. \synopsis optional &<a href="/thrust/api/classes/classoptional.html#function-operator=">operator=(optional<U> && rhs)</a>; 

<h3 id="function-emplace">
Function <code>optional::&gt;::emplace</code>
</h3>

<code class="doxybook">
<span>template &lt;class... Args&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ T & </span><span><b>emplace</b>(Args &&... args);</span></code>
Constructs the value in-place, destroying the current one if there is one. 

\group emplace 

<h3 id="function-emplace">
Function <code>optional::&gt;::emplace</code>
</h3>

<code class="doxybook">
<span>template &lt;class U,</span>
<span>&nbsp;&nbsp;class... Args&gt;</span>
<span>__thrust_exec_check_disable__ __host__ __device__ detail::enable_if_t< std::is_constructible< T, std::initializer_list< U > &, Args &&... >::<a href="/thrust/api/classes/classoptional.html#function-value">value</a>, T & > </span><span><b>emplace</b>(std::initializer_list< U > il,</span>
<span>&nbsp;&nbsp;Args &&... args);</span></code>
\group emplace \synopsis template <class U, class... 

Args>\nT& <a href="/thrust/api/classes/classoptional.html#function-emplace">emplace(std::initializer_list<U> il, Args &&... args)</a>; 

<h3 id="function-swap">
Function <code>optional::&gt;::swap</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ __device__ void </span><span><b>swap</b>(<a href="/thrust/api/classes/classoptional.html">optional</a> & rhs);</span></code>
Swaps this optional with the other. 

If neither optionals have a value, nothing happens. If both have a value, the values are swapped. If one has a value, it is moved to the other and the movee is left valueless. 

<h3 id="function-operator->">
Function <code>optional::&gt;::operator-&gt;</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr const __device__ T * </span><span><b>operator-></b>() const;</span></code>
**Returns**:
a pointer to the stored value \requires a value is stored \group pointer \synopsis constexpr const T *operator->() const; 

<h3 id="function-operator->">
Function <code>optional::&gt;::operator-&gt;</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ T * </span><span><b>operator-></b>();</span></code>
\group pointer \synopsis constexpr T *operator->(); 

<h3 id="function-operator*">
Function <code>optional::&gt;::operator&#42;</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ T & </span><span><b>operator*</b>();</span></code>
**Returns**:
the stored value \requires a value is stored \group deref \synopsis constexpr T &<a href="/thrust/api/classes/classoptional.html#function-operator*">operator*()</a>; 

<h3 id="function-operator*">
Function <code>optional::&gt;::operator&#42;</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr const __device__ T & </span><span><b>operator*</b>() const;</span></code>
\group deref \synopsis constexpr const T &operator*() const; 

<h3 id="function-operator*">
Function <code>optional::&gt;::operator&#42;</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ T && </span><span><b>operator*</b>();</span></code>
\exclude 

<h3 id="function-operator*">
Function <code>optional::&gt;::operator&#42;</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr const __device__ T && </span><span><b>operator*</b>() const;</span></code>
\exclude 

<h3 id="function-has_value">
Function <code>optional::&gt;::has&#95;value</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ bool </span><span><b>has_value</b>() const;</span></code>
**Returns**:
whether or not the optional has a value \group has_value 

<h3 id="function-operator bool">
Function <code>optional::bool</code>
</h3>

<code class="doxybook">
<span>explicit __thrust_exec_check_disable__ __host__ constexpr __device__ </span><span><b>operator bool</b>() const;</span></code>
\group has_value 

<h3 id="function-value">
Function <code>optional::&gt;::value</code>
</h3>

<code class="doxybook">
<span>constexpr __host__ T & </span><span><b>value</b>();</span></code>
**Returns**:
the contained value if there is one, otherwise throws [bad_optional_access] \group value \synopsis constexpr T &<a href="/thrust/api/classes/classoptional.html#function-value">value()</a>; 

<h3 id="function-value">
Function <code>optional::&gt;::value</code>
</h3>

<code class="doxybook">
<span>constexpr const __host__ T & </span><span><b>value</b>() const;</span></code>
\group value \synopsis constexpr const T &value() const; 

<h3 id="function-value">
Function <code>optional::&gt;::value</code>
</h3>

<code class="doxybook">
<span>constexpr __host__ T && </span><span><b>value</b>();</span></code>
\exclude 

<h3 id="function-value">
Function <code>optional::&gt;::value</code>
</h3>

<code class="doxybook">
<span>constexpr const __host__ T && </span><span><b>value</b>() const;</span></code>
\exclude 

<h3 id="function-value_or">
Function <code>optional::&gt;::value&#95;or</code>
</h3>

<code class="doxybook">
<span>template &lt;class U&gt;</span>
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ T </span><span><b>value_or</b>(U && u) const;</span></code>
**Returns**:
the stored value if there is one, otherwise returns <code>u</code> \group value_or 

<h3 id="function-value_or">
Function <code>optional::&gt;::value&#95;or</code>
</h3>

<code class="doxybook">
<span>template &lt;class U&gt;</span>
<span>__thrust_exec_check_disable__ __host__ constexpr __device__ T </span><span><b>value_or</b>(U && u);</span></code>
\group value_or 

<h3 id="function-reset">
Function <code>optional::&gt;::reset</code>
</h3>

<code class="doxybook">
<span>__thrust_exec_check_disable__ __host__ __device__ void </span><span><b>reset</b>();</span></code>
Destroys the stored value if one exists, making the optional empty. 


