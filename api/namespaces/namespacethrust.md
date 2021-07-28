---
title: thrust
nav_exclude: true
has_children: true
has_toc: false
---

# Namespace `thrust`

<code class="doxybook">
<span>namespace thrust {</span>
<br>
<span class="doxybook-comment">/* <code>thrust::cpp</code> is a top-level alias for thrust::system::cpp.  */</span><span>namespace <b><a href="/api/namespaces/namespacethrust_1_1cpp.html">thrust::cpp</a></b> { <i>…</i> }</span>
<br>
<span class="doxybook-comment">/* <code>thrust::cuda</code> is a top-level alias for <code>thrust::system::cuda</code>.  */</span><span>namespace <b><a href="/api/namespaces/namespacethrust_1_1cuda.html">thrust::cuda</a></b> { <i>…</i> }</span>
<br>
<span class="doxybook-comment">/* <code>thrust::omp</code> is a top-level alias for thrust::system::omp.  */</span><span>namespace <b><a href="/api/namespaces/namespacethrust_1_1omp.html">thrust::omp</a></b> { <i>…</i> }</span>
<br>
<span class="doxybook-comment">/* Facilities for constructing simple functions inline.  */</span><span>namespace <b><a href="/api/namespaces/namespacethrust_1_1placeholders.html">thrust::placeholders</a></b> { <i>…</i> }</span>
<br>
<span class="doxybook-comment">/* <code>thrust::random</code> is the namespace which contains random number engine class templates, random number engine adaptor class templates, engines with predefined parameters, and random number distribution class templates. They are provided in a separate namespace for import convenience but are also aliased in the top-level <code>thrust</code> namespace for easy access.  */</span><span>namespace <b><a href="/api/namespaces/namespacethrust_1_1random.html">thrust::random</a></b> { <i>…</i> }</span>
<br>
<span class="doxybook-comment">/* <code>thrust::system</code> is the namespace which contains specific Thrust backend systems. It also contains functionality for reporting error conditions originating from the operating system or other low-level application program interfaces such as the CUDA runtime. They are provided in a separate namespace for import convenience but are also aliased in the top-level <code>thrust</code> namespace for easy access.  */</span><span>namespace <b><a href="/api/namespaces/namespacethrust_1_1system.html">thrust::system</a></b> { <i>…</i> }</span>
<br>
<span class="doxybook-comment">/* <code>thrust::tbb</code> is a top-level alias for thrust::system::tbb.  */</span><span>namespace <b><a href="/api/namespaces/namespacethrust_1_1tbb.html">thrust::tbb</a></b> { <i>…</i> }</span>
<br>
<span class="doxybook-comment">/* <code>test&#95;namespace</code> is a namespace intended to exercise and test Doxybook rendering.  */</span><span>namespace <b><a href="/api/namespaces/namespacethrust_1_1test__namespace.html">thrust::test&#95;namespace</a></b> { <i>…</i> }</span>
<br>
<span class="doxybook-comment">/* <code>test&#95;class</code> is a class intended to exercise and test Doxybook rendering.  */</span><span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename U&gt;</span>
<span>class <b><a href="/api/classes/classthrust_1_1test__class.html">test&#95;class</a></b>;</span>
<br>
<span class="doxybook-comment">/* <code>test&#95;derived&#95;class</code> is a derived class intended to exercise and test Doxybook rendering.  */</span><span>class <b><a href="/api/classes/classthrust_1_1test__derived__class.html">test&#95;derived&#95;class</a></b>;</span>
<br>
<span class="doxybook-comment">/* <code>test&#95;predefined&#95;friend&#95;struct</code> is a class intended to exercise and test Doxybook rendering.  */</span><span>template &lt;typename... Z&gt;</span>
<span>struct <b><a href="/api/classes/structthrust_1_1test__predefined__friend__struct.html">test&#95;predefined&#95;friend&#95;struct</a></b>;</span>
<br>
<span class="doxybook-comment">/* <code>test&#95;enum</code> is an enum namespace intended to exercise and test Doxybook rendering.  */</span><span>enum class <b><a href="/api/groups/group__test.html#enum-test_enum">test&#95;enum</a></b>;</span>
<br>
<span class="doxybook-comment">/* <code>test&#95;alias</code> is a type alias intended to exercise and test Doxybook rendering.  */</span><span>using <b><a href="/api/groups/group__test.html#using-test_alias">test&#95;alias</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>test&#95;predefined&#95;friend&#95;function</code> is a function intended to exercise and test Doxybook rendering.  */</span><span>template &lt;typename Z&gt;</span>
<span>void </span><span><b><a href="/api/groups/group__test.html#function-test_predefined_friend_function">test&#95;predefined&#95;friend&#95;function</a></b>();</span>
<br>
<span class="doxybook-comment">/* <code>test&#95;function</code> is a function intended to exercise and test Doxybook rendering.  */</span><span>template &lt;typename T&gt;</span>
<span>void </span><span><b><a href="/api/groups/group__test.html#function-test_function">test&#95;function</a></b>(T const & a,</span>
<span>&nbsp;&nbsp;<a href="/api/classes/classthrust_1_1test__class.html">test_class</a>< T, T const > && b);</span>
<br>
<span class="doxybook-comment">/* <code>test&#95;parameter&#95;overflow&#95;function</code> is a function intended to test Doxybook's rendering of function and template parameters that exceed the length of a line.  */</span><span>template &lt;typename T = test&#95;predefined&#95;friend&#95;struct&lt;int, int, int, int, int, int, int, int, int, int, int, int&gt;,</span>
<span>&nbsp;&nbsp;typename U = test&#95;predefined&#95;friend&#95;struct&lt;int, int, int, int, int, int, int, int, int, int, int, int&gt;,</span>
<span>&nbsp;&nbsp;typename V = test&#95;predefined&#95;friend&#95;struct&lt;int, int, int, int, int, int, int, int, int, int, int, int&gt;&gt;</span>
<span><a href="/api/classes/structthrust_1_1test__predefined__friend__struct.html">test_predefined_friend_struct</a>< int, int, int, int, int, int, int, int, int, int, int, int, int, int, int > </span><span><b><a href="/api/groups/group__test.html#function-test_parameter_overflow_function">test&#95;parameter&#95;overflow&#95;function</a></b>(<a href="/api/classes/structthrust_1_1test__predefined__friend__struct.html">test_predefined_friend_struct</a>< int, int, int, int, int, int, int, int, int, int, int, int > t,</span>
<span>&nbsp;&nbsp;<a href="/api/classes/structthrust_1_1test__predefined__friend__struct.html">test_predefined_friend_struct</a>< int, int, int, int, int, int, int, int, int, int, int, int > u,</span>
<span>&nbsp;&nbsp;<a href="/api/classes/structthrust_1_1test__predefined__friend__struct.html">test_predefined_friend_struct</a>< int, int, int, int, int, int, int, int, int, int, int, int > v);</span>
<span>} /* namespace thrust */</span>
</code>

## Member Classes

<h3 id="class-thrust::test_class">
<a href="/api/classes/classthrust_1_1test__class.html">Class <code>thrust::test&#95;class</code>
</a>
</h3>

<code>test&#95;class</code> is a class intended to exercise and test Doxybook rendering. 

<h3 id="class-thrust::test_derived_class">
<a href="/api/classes/classthrust_1_1test__derived__class.html">Class <code>thrust::test&#95;derived&#95;class</code>
</a>
</h3>

<code>test&#95;derived&#95;class</code> is a derived class intended to exercise and test Doxybook rendering. 

**Inherits From**:
[`thrust::test_class< int, double >`](/api/classes/classthrust_1_1test__class.html)

<h3 id="struct-thrust::test_predefined_friend_struct">
<a href="/api/classes/structthrust_1_1test__predefined__friend__struct.html">Struct <code>thrust::test&#95;predefined&#95;friend&#95;struct</code>
</a>
</h3>

<code>test&#95;predefined&#95;friend&#95;struct</code> is a class intended to exercise and test Doxybook rendering. 


## Types

<h3 id="enum-test_enum">
Enum Class <code>test&#95;enum</code>
</h3>

| Enumerator | Value | Description |
|------------|-------|-------------|
| `X` | `1` | An enumerator. It is equal to 1.  |
| `Y` | `X` |  |
| `Z` | `2` |  |

<code>test&#95;enum</code> is an enum namespace intended to exercise and test Doxybook rendering. 

<h3 id="using-test_alias">
Type Alias <code>test&#95;alias</code>
</h3>

<code class="doxybook">
<span>using <b>test_alias</b> = &lt;a href="/api/classes/classthrust&#95;1&#95;1test&#95;&#95;class.html"&gt;test&#95;class&lt;/a&gt;&lt; int, double &gt;;</span></code>
<code>test&#95;alias</code> is a type alias intended to exercise and test Doxybook rendering. 


## Functions

<h3 id="function-test_predefined_friend_function">
Function <code>thrust::test&#95;predefined&#95;friend&#95;function</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Z&gt;</span>
<span>void </span><span><b>test_predefined_friend_function</b>();</span></code>
<code>test&#95;predefined&#95;friend&#95;function</code> is a function intended to exercise and test Doxybook rendering. 

<h3 id="function-test_function">
Function <code>thrust::test&#95;function</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>void </span><span><b>test_function</b>(T const & a,</span>
<span>&nbsp;&nbsp;<a href="/api/classes/classthrust_1_1test__class.html">test_class</a>< T, T const > && b);</span></code>
<code>test&#95;function</code> is a function intended to exercise and test Doxybook rendering. 

<h3 id="function-test_parameter_overflow_function">
Function <code>thrust::test&#95;parameter&#95;overflow&#95;function</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T = test&#95;predefined&#95;friend&#95;struct&lt;int, int, int, int, int, int, int, int, int, int, int, int&gt;,</span>
<span>&nbsp;&nbsp;typename U = test&#95;predefined&#95;friend&#95;struct&lt;int, int, int, int, int, int, int, int, int, int, int, int&gt;,</span>
<span>&nbsp;&nbsp;typename V = test&#95;predefined&#95;friend&#95;struct&lt;int, int, int, int, int, int, int, int, int, int, int, int&gt;&gt;</span>
<span><a href="/api/classes/structthrust_1_1test__predefined__friend__struct.html">test_predefined_friend_struct</a>< int, int, int, int, int, int, int, int, int, int, int, int, int, int, int > </span><span><b>test_parameter_overflow_function</b>(<a href="/api/classes/structthrust_1_1test__predefined__friend__struct.html">test_predefined_friend_struct</a>< int, int, int, int, int, int, int, int, int, int, int, int > t,</span>
<span>&nbsp;&nbsp;<a href="/api/classes/structthrust_1_1test__predefined__friend__struct.html">test_predefined_friend_struct</a>< int, int, int, int, int, int, int, int, int, int, int, int > u,</span>
<span>&nbsp;&nbsp;<a href="/api/classes/structthrust_1_1test__predefined__friend__struct.html">test_predefined_friend_struct</a>< int, int, int, int, int, int, int, int, int, int, int, int > v);</span></code>
<code>test&#95;parameter&#95;overflow&#95;function</code> is a function intended to test Doxybook's rendering of function and template parameters that exceed the length of a line. 


