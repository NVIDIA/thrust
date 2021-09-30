---
title: Arithmetic Operations
parent: Predefined Function Objects
grand_parent: Function Objects
nav_exclude: false
has_children: true
has_toc: false
---

# Arithmetic Operations

<code class="doxybook">
<span>template &lt;typename T = void&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structplus.html">plus</a></b>;</span>
<br>
<span>struct <b><a href="/thrust/api/classes/structplus_3_01void_01_4.html">plus&lt; void &gt;</a></b>;</span>
<br>
<span>template &lt;typename T = void&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structminus.html">minus</a></b>;</span>
<br>
<span>struct <b><a href="/thrust/api/classes/structminus_3_01void_01_4.html">minus&lt; void &gt;</a></b>;</span>
<br>
<span>template &lt;typename T = void&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structmultiplies.html">multiplies</a></b>;</span>
<br>
<span>struct <b><a href="/thrust/api/classes/structmultiplies_3_01void_01_4.html">multiplies&lt; void &gt;</a></b>;</span>
<br>
<span>template &lt;typename T = void&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structdivides.html">divides</a></b>;</span>
<br>
<span>struct <b><a href="/thrust/api/classes/structdivides_3_01void_01_4.html">divides&lt; void &gt;</a></b>;</span>
<br>
<span>template &lt;typename T = void&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structmodulus.html">modulus</a></b>;</span>
<br>
<span>struct <b><a href="/thrust/api/classes/structmodulus_3_01void_01_4.html">modulus&lt; void &gt;</a></b>;</span>
<br>
<span>template &lt;typename T = void&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structnegate.html">negate</a></b>;</span>
<br>
<span>struct <b><a href="/thrust/api/classes/structnegate_3_01void_01_4.html">negate&lt; void &gt;</a></b>;</span>
<br>
<span>template &lt;typename T = void&gt;</span>
<span>struct <b><a href="/thrust/api/classes/structsquare.html">square</a></b>;</span>
<br>
<span>struct <b><a href="/thrust/api/classes/structsquare_3_01void_01_4.html">square&lt; void &gt;</a></b>;</span>
<br>
<span>#define <b><a href="/thrust/api/groups/group__arithmetic__operations.html#define-thrust_unary_functor_void_specialization">THRUST&#95;UNARY&#95;FUNCTOR&#95;VOID&#95;SPECIALIZATION</a></b> = <i>see below</i>;</span>
<br>
<span>#define <b><a href="/thrust/api/groups/group__arithmetic__operations.html#define-thrust_binary_functor_void_specialization">THRUST&#95;BINARY&#95;FUNCTOR&#95;VOID&#95;SPECIALIZATION</a></b> = <i>see below</i>;</span>
<br>
<span>#define <b><a href="/thrust/api/groups/group__arithmetic__operations.html#define-thrust_binary_functor_void_specialization_op">THRUST&#95;BINARY&#95;FUNCTOR&#95;VOID&#95;SPECIALIZATION&#95;OP</a></b> = <i>see below</i>;</span>
</code>

## Member Classes

<h3 id="struct-plus">
<a href="/thrust/api/classes/structplus.html">Struct <code>plus</code>
</a>
</h3>

<h3 id="struct-plus< void >">
<a href="/thrust/api/classes/structplus_3_01void_01_4.html">Struct <code>plus&lt; void &gt;</code>
</a>
</h3>

<h3 id="struct-minus">
<a href="/thrust/api/classes/structminus.html">Struct <code>minus</code>
</a>
</h3>

<h3 id="struct-minus< void >">
<a href="/thrust/api/classes/structminus_3_01void_01_4.html">Struct <code>minus&lt; void &gt;</code>
</a>
</h3>

<h3 id="struct-multiplies">
<a href="/thrust/api/classes/structmultiplies.html">Struct <code>multiplies</code>
</a>
</h3>

<h3 id="struct-multiplies< void >">
<a href="/thrust/api/classes/structmultiplies_3_01void_01_4.html">Struct <code>multiplies&lt; void &gt;</code>
</a>
</h3>

<h3 id="struct-divides">
<a href="/thrust/api/classes/structdivides.html">Struct <code>divides</code>
</a>
</h3>

<h3 id="struct-divides< void >">
<a href="/thrust/api/classes/structdivides_3_01void_01_4.html">Struct <code>divides&lt; void &gt;</code>
</a>
</h3>

<h3 id="struct-modulus">
<a href="/thrust/api/classes/structmodulus.html">Struct <code>modulus</code>
</a>
</h3>

<h3 id="struct-modulus< void >">
<a href="/thrust/api/classes/structmodulus_3_01void_01_4.html">Struct <code>modulus&lt; void &gt;</code>
</a>
</h3>

<h3 id="struct-negate">
<a href="/thrust/api/classes/structnegate.html">Struct <code>negate</code>
</a>
</h3>

<h3 id="struct-negate< void >">
<a href="/thrust/api/classes/structnegate_3_01void_01_4.html">Struct <code>negate&lt; void &gt;</code>
</a>
</h3>

<h3 id="struct-square">
<a href="/thrust/api/classes/structsquare.html">Struct <code>square</code>
</a>
</h3>

<h3 id="struct-square< void >">
<a href="/thrust/api/classes/structsquare_3_01void_01_4.html">Struct <code>square&lt; void &gt;</code>
</a>
</h3>


## Macros

<h3 id="define-THRUST_UNARY_FUNCTOR_VOID_SPECIALIZATION">
Define <code>THRUST&#95;UNARY&#95;FUNCTOR&#95;VOID&#95;SPECIALIZATION</code>
</h3>

<code class="doxybook">
  <span>#define <b>THRUST_UNARY_FUNCTOR_VOID_SPECIALIZATION</b>   template &lt;&gt;                                                                  \
  struct func&lt;void&gt;                                                            \
  {                                                                            \
    using is&#95;transparent = void;                                               \
    &#95;&#95;thrust&#95;exec&#95;check&#95;disable&#95;&#95;                                              \
    template &lt;typename T&gt;                                                      \
    &#95;&#95;host&#95;&#95; &#95;&#95;device&#95;&#95;                                                        \
    constexpr auto operator()(T&& x) const                                     \
      noexcept(noexcept(impl)) THRUST&#95;TRAILING&#95;RETURN(decltype(impl))          \
    {                                                                          \
      return impl;                                                             \
    }                                                                          \
  };</span></code>
<h3 id="define-THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION">
Define <code>THRUST&#95;BINARY&#95;FUNCTOR&#95;VOID&#95;SPECIALIZATION</code>
</h3>

<code class="doxybook">
  <span>#define <b>THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION</b>   template &lt;&gt;                                                                  \
  struct func&lt;void&gt;                                                            \
  {                                                                            \
    using is&#95;transparent = void;                                               \
    &#95;&#95;thrust&#95;exec&#95;check&#95;disable&#95;&#95;                                              \
    template &lt;typename T1, typename T2&gt;                                        \
    &#95;&#95;host&#95;&#95; &#95;&#95;device&#95;&#95;                                                        \
    constexpr auto operator()(T1&& t1, T2&& t2) const                          \
      noexcept(noexcept(impl)) THRUST&#95;TRAILING&#95;RETURN(decltype(impl))          \
    {                                                                          \
      return impl;                                                             \
    }                                                                          \
  };</span></code>
<h3 id="define-THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION_OP">
Define <code>THRUST&#95;BINARY&#95;FUNCTOR&#95;VOID&#95;SPECIALIZATION&#95;OP</code>
</h3>

<code class="doxybook">
  <span>#define <b>THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION_OP</b>   THRUST&#95;BINARY&#95;FUNCTOR&#95;VOID&#95;SPECIALIZATION(                                   \
    func, THRUST&#95;FWD(t1) op THRUST&#95;FWD(t2));</span></code>

