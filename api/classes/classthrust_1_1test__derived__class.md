---
title: thrust::test_derived_class
summary: test_derived_class is a derived class intended to exercise and test Doxybook rendering. 
parent: Test
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::test_derived_class`

<code class="doxybook">
<span>#include <thrust/doxybook_test.h></span><br>
<span>class thrust::test&#95;derived&#95;class {</span>
<span>public:</span><span class="doxybook-comment">/* Inherited from <code><b><a href="/api/classes/classthrust_1_1test__class.html">thrust::test&#95;class&lt; int, double &gt;</a></b></code> */</span><span>&nbsp;&nbsp;enum class <b><a href="/api/classes/classthrust_1_1test__class.html#enum-test_enum_class">test&#95;enum&#95;class</a></b>;</span>
<span class="doxybook-comment">/* Inherited from <code><b><a href="/api/classes/classthrust_1_1test__class.html">thrust::test&#95;class&lt; int, double &gt;</a></b></code> */</span><br>
<span>&nbsp;&nbsp;template &lt;typename X,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typename Y&gt;</span>
<span>&nbsp;&nbsp;using <b><a href="/api/classes/classthrust_1_1test__class.html#using-test_type_alias">test&#95;type&#95;alias</a></b> = <i>see below</i>;</span>
<span class="doxybook-comment">/* Inherited from <code><b><a href="/api/classes/classthrust_1_1test__class.html">thrust::test&#95;class&lt; int, double &gt;</a></b></code> */</span><br>
<span>&nbsp;&nbsp;template &lt;typename Z&gt;</span>
<span>&nbsp;&nbsp;struct <b><a href="/api/classes/structthrust_1_1test__class_1_1test__nested__struct.html">test&#95;nested&#95;struct</a></b>;</span>
<span class="doxybook-comment">/* Inherited from <code><b><a href="/api/classes/classthrust_1_1test__class.html">thrust::test&#95;class&lt; int, double &gt;</a></b></code> */</span><br>
<span>&nbsp;&nbsp;template &lt;typename Z&gt;</span>
<span>&nbsp;&nbsp;friend class <b>test&#95;friend&#95;class</b>;</span>
<span class="doxybook-comment">/* Inherited from <code><b><a href="/api/classes/classthrust_1_1test__class.html">thrust::test&#95;class&lt; int, double &gt;</a></b></code> */</span><br>
<span>&nbsp;&nbsp;template &lt;typename... Z&gt;</span>
<span>&nbsp;&nbsp;friend struct <b>thrust::test&#95;predefined&#95;friend&#95;struct</b>;</span>
<span class="doxybook-comment">/* Inherited from <code><b><a href="/api/classes/classthrust_1_1test__class.html">thrust::test&#95;class&lt; int, double &gt;</a></b></code> */</span><br>
<span class="doxybook-comment">&nbsp;&nbsp;/* A test member variable.  */</span><span>&nbsp;&nbsp;int <b><a href="/api/classes/classthrust_1_1test__class.html#variable-test_member_variable">test&#95;member&#95;variable</a></b> = <i>see below</i>;</span>
<span class="doxybook-comment">/* Inherited from <code><b><a href="/api/classes/classthrust_1_1test__class.html">thrust::test&#95;class&lt; int, double &gt;</a></b></code> */</span><br>
<span class="doxybook-comment">&nbsp;&nbsp;/* A test member constant.  */</span><span>&nbsp;&nbsp;constexpr int <b><a href="/api/classes/classthrust_1_1test__class.html#variable-test_member_constant">test&#95;member&#95;constant</a></b> = <i>see below</i>;</span>
<span class="doxybook-comment">/* Inherited from <code><b><a href="/api/classes/classthrust_1_1test__class.html">thrust::test&#95;class&lt; int, double &gt;</a></b></code> */</span><br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Construct an empty test class.  */</span><span>&nbsp;&nbsp;<b><a href="/api/classes/classthrust_1_1test__class.html#function-test_class">test&#95;class</a></b>() = default;</span>
<span class="doxybook-comment">/* Inherited from <code><b><a href="/api/classes/classthrust_1_1test__class.html">thrust::test&#95;class&lt; int, double &gt;</a></b></code> */</span><br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Construct a test class.  */</span><span>&nbsp;&nbsp;__host__ constexpr __device__ </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classthrust_1_1test__class.html#function-test_class">test&#95;class</a></b>(int);</span>
<span class="doxybook-comment">/* Inherited from <code><b><a href="/api/classes/classthrust_1_1test__class.html">thrust::test&#95;class&lt; int, double &gt;</a></b></code> */</span><br>
<span class="doxybook-comment">&nbsp;&nbsp;/* <code>test&#95;member&#95;function</code> is a function intended to exercise and test Doxybook rendering.  */</span><span>&nbsp;&nbsp;virtual __host__ constexpr virtual __device__ int </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classthrust_1_1test__class.html#function-test_member_function">test&#95;member&#95;function</a></b>() = 0;</span>
<span class="doxybook-comment">/* Inherited from <code><b><a href="/api/classes/classthrust_1_1test__class.html">thrust::test&#95;class&lt; int, double &gt;</a></b></code> */</span><br>
<span class="doxybook-comment">&nbsp;&nbsp;/* <code>test&#95;parameter&#95;overflow&#95;member&#95;function</code> is a function intended to test Doxybook's rendering of function and template parameters that exceed the length of a line.  */</span><span>&nbsp;&nbsp;template &lt;typename T = test&#95;predefined&#95;friend&#95;struct&lt;int, int, int, int, int, int, int, int, int, int, int, int&gt;,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typename U = test&#95;predefined&#95;friend&#95;struct&lt;int, int, int, int, int, int, int, int, int, int, int, int&gt;,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typename V = test&#95;predefined&#95;friend&#95;struct&lt;int, int, int, int, int, int, int, int, int, int, int, int&gt;&gt;</span>
<span>&nbsp;&nbsp;<a href="/api/classes/structthrust_1_1test__predefined__friend__struct.html">test_predefined_friend_struct</a>< int, int, int, int, int, int, int, int, int, int, int, int, int, int, int > </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classthrust_1_1test__class.html#function-test_parameter_overflow_member_function">test&#95;parameter&#95;overflow&#95;member&#95;function</a></b>(<a href="/api/classes/structthrust_1_1test__predefined__friend__struct.html">test_predefined_friend_struct</a>< int, int, int, int, int, int, int, int, int, int, int, int, int, int, int > t,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;<a href="/api/classes/structthrust_1_1test__predefined__friend__struct.html">test_predefined_friend_struct</a>< int, int, int, int, int, int, int, int, int, int, int, int, int, int, int > u,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;<a href="/api/classes/structthrust_1_1test__predefined__friend__struct.html">test_predefined_friend_struct</a>< int, int, int, int, int, int, int, int, int, int, int, int, int, int, int > v);</span>
<span class="doxybook-comment">/* Inherited from <code><b><a href="/api/classes/classthrust_1_1test__class.html">thrust::test&#95;class&lt; int, double &gt;</a></b></code> */</span><br>
<span>&nbsp;&nbsp;template &lt;typename Z&gt;</span>
<span>&nbsp;&nbsp;friend void </span>
<span>&nbsp;&nbsp;<b>test&#95;friend&#95;function</b>();</span>
<span class="doxybook-comment">/* Inherited from <code><b><a href="/api/classes/classthrust_1_1test__class.html">thrust::test&#95;class&lt; int, double &gt;</a></b></code> */</span><br>
<span class="doxybook-comment">&nbsp;&nbsp;/* <code>test&#95;predefined&#95;friend&#95;function</code> is a function intended to exercise and test Doxybook rendering.  */</span><span>&nbsp;&nbsp;template &lt;typename Z&gt;</span>
<span>&nbsp;&nbsp;friend void </span>
<span>&nbsp;&nbsp;<b>test&#95;predefined&#95;friend&#95;function</b>();</span>
<br>
<span>protected:</span><span class="doxybook-comment">/* Inherited from <code><b><a href="/api/classes/classthrust_1_1test__class.html">thrust::test&#95;class&lt; int, double &gt;</a></b></code> */</span><span>&nbsp;&nbsp;template &lt;typename Z&gt;</span>
<span>&nbsp;&nbsp;class <b><a href="/api/classes/classthrust_1_1test__class_1_1test__protected__nested__class.html">test&#95;protected&#95;nested&#95;class</a></b>;</span>
<span class="doxybook-comment">/* Inherited from <code><b><a href="/api/classes/classthrust_1_1test__class.html">thrust::test&#95;class&lt; int, double &gt;</a></b></code> */</span><br>
<span class="doxybook-comment">&nbsp;&nbsp;/* <code>test&#95;protected&#95;member&#95;function</code> is a function intended to exercise and test Doxybook rendering.  */</span><span>&nbsp;&nbsp;__device__ auto </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classthrust_1_1test__class.html#function-test_protected_member_function">test&#95;protected&#95;member&#95;function</a></b>();</span>
<span>};</span>
</code>

