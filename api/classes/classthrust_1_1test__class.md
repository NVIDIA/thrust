---
title: thrust::test_class
summary: test_class is a class intended to exercise and test Doxybook rendering. 
parent: Test
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::test_class`

<code>test&#95;class</code> is a class intended to exercise and test Doxybook rendering. 

It does many things.

**See**:
test_function 

<code class="doxybook">
<span>#include <thrust/doxybook_test.h></span><br>
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;typename U&gt;</span>
<span>class thrust::test&#95;class {</span>
<span>public:</span><span>&nbsp;&nbsp;enum class <b><a href="/api/classes/classthrust_1_1test__class.html#enum-test_enum_class">test&#95;enum&#95;class</a></b>;</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename X,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typename Y&gt;</span>
<span>&nbsp;&nbsp;using <b><a href="/api/classes/classthrust_1_1test__class.html#using-test_type_alias">test&#95;type&#95;alias</a></b> = <i>see below</i>;</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename Z&gt;</span>
<span>&nbsp;&nbsp;struct <b><a href="/api/classes/structthrust_1_1test__class_1_1test__nested__struct.html">test&#95;nested&#95;struct</a></b>;</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename Z&gt;</span>
<span>&nbsp;&nbsp;friend class <b>test&#95;friend&#95;class</b>;</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename... Z&gt;</span>
<span>&nbsp;&nbsp;friend struct <b>thrust::test&#95;predefined&#95;friend&#95;struct</b>;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* A test member variable.  */</span><span>&nbsp;&nbsp;int <b><a href="/api/classes/classthrust_1_1test__class.html#variable-test_member_variable">test&#95;member&#95;variable</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* A test member constant.  */</span><span>&nbsp;&nbsp;constexpr int <b><a href="/api/classes/classthrust_1_1test__class.html#variable-test_member_constant">test&#95;member&#95;constant</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Construct an empty test class.  */</span><span>&nbsp;&nbsp;<b><a href="/api/classes/classthrust_1_1test__class.html#function-test_class">test&#95;class</a></b>() = default;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* Construct a test class.  */</span><span>&nbsp;&nbsp;__host__ constexpr __device__ </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classthrust_1_1test__class.html#function-test_class">test&#95;class</a></b>(int);</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* <code>test&#95;member&#95;function</code> is a function intended to exercise and test Doxybook rendering.  */</span><span>&nbsp;&nbsp;virtual __host__ constexpr virtual __device__ int </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classthrust_1_1test__class.html#function-test_member_function">test&#95;member&#95;function</a></b>() = 0;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* <code>test&#95;parameter&#95;overflow&#95;member&#95;function</code> is a function intended to test Doxybook's rendering of function and template parameters that exceed the length of a line.  */</span><span>&nbsp;&nbsp;template &lt;typename T = test&#95;predefined&#95;friend&#95;struct&lt;int, int, int, int, int, int, int, int, int, int, int, int&gt;,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typename U = test&#95;predefined&#95;friend&#95;struct&lt;int, int, int, int, int, int, int, int, int, int, int, int&gt;,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typename V = test&#95;predefined&#95;friend&#95;struct&lt;int, int, int, int, int, int, int, int, int, int, int, int&gt;&gt;</span>
<span>&nbsp;&nbsp;<a href="/api/classes/structthrust_1_1test__predefined__friend__struct.html">test_predefined_friend_struct</a>< int, int, int, int, int, int, int, int, int, int, int, int, int, int, int > </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classthrust_1_1test__class.html#function-test_parameter_overflow_member_function">test&#95;parameter&#95;overflow&#95;member&#95;function</a></b>(<a href="/api/classes/structthrust_1_1test__predefined__friend__struct.html">test_predefined_friend_struct</a>< int, int, int, int, int, int, int, int, int, int, int, int, int, int, int > t,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;<a href="/api/classes/structthrust_1_1test__predefined__friend__struct.html">test_predefined_friend_struct</a>< int, int, int, int, int, int, int, int, int, int, int, int, int, int, int > u,</span>
<span>&nbsp;&nbsp;&nbsp;&nbsp;<a href="/api/classes/structthrust_1_1test__predefined__friend__struct.html">test_predefined_friend_struct</a>< int, int, int, int, int, int, int, int, int, int, int, int, int, int, int > v);</span>
<br>
<span>&nbsp;&nbsp;template &lt;typename Z&gt;</span>
<span>&nbsp;&nbsp;friend void </span>
<span>&nbsp;&nbsp;<b>test&#95;friend&#95;function</b>();</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* <code>test&#95;predefined&#95;friend&#95;function</code> is a function intended to exercise and test Doxybook rendering.  */</span><span>&nbsp;&nbsp;template &lt;typename Z&gt;</span>
<span>&nbsp;&nbsp;friend void </span>
<span>&nbsp;&nbsp;<b>test&#95;predefined&#95;friend&#95;function</b>();</span>
<br>
<span>protected:</span><span>&nbsp;&nbsp;template &lt;typename Z&gt;</span>
<span>&nbsp;&nbsp;class <b><a href="/api/classes/classthrust_1_1test__class_1_1test__protected__nested__class.html">test&#95;protected&#95;nested&#95;class</a></b>;</span>
<br>
<span class="doxybook-comment">&nbsp;&nbsp;/* <code>test&#95;protected&#95;member&#95;function</code> is a function intended to exercise and test Doxybook rendering.  */</span><span>&nbsp;&nbsp;__device__ auto </span><span>&nbsp;&nbsp;<b><a href="/api/classes/classthrust_1_1test__class.html#function-test_protected_member_function">test&#95;protected&#95;member&#95;function</a></b>();</span>
<span>};</span>
</code>

## Member Classes

<h3 id="struct-thrust::test_class::test_nested_struct">
<a href="/api/classes/structthrust_1_1test__class_1_1test__nested__struct.html">Struct <code>thrust::test&#95;class::thrust::test&#95;class::test&#95;nested&#95;struct</code>
</a>
</h3>


## Member Types

<h3 id="enum-test_enum_class">
Enum Class <code>thrust::test&#95;class::test&#95;enum&#95;class</code>
</h3>

| Enumerator | Value | Description |
|------------|-------|-------------|
| `A` | `15` | An enumerator. It is equal to 15.  |
| `B` |  |  |
| `C` |  |  |

<h3 id="using-test_type_alias">
Type Alias <code>thrust::test&#95;class::test&#95;type&#95;alias</code>
</h3>

<code class="doxybook">
<span>template &lt;typename X,</span>
<span>&nbsp;&nbsp;typename Y&gt;</span>
<span>using <b>test_type_alias</b> = &lt;a href="/api/classes/classthrust&#95;1&#95;1test&#95;&#95;class.html"&gt;test&#95;class&lt;/a&gt;&lt; X, Y &gt;;</span></code>

## Member Variables

<h3 id="variable-test_member_variable">
Variable <code>thrust::test&#95;class::thrust::test&#95;class::test&#95;member&#95;variable</code>
</h3>

<code class="doxybook">
<span>int <b>test_member_variable</b> = 0;</span></code>
A test member variable. 

<h3 id="variable-test_member_constant">
Variable <code>thrust::test&#95;class::thrust::test&#95;class::test&#95;member&#95;constant</code>
</h3>

<code class="doxybook">
<span>constexpr int <b>test_member_constant</b> = 42;</span></code>
A test member constant. 


## Member Functions

<h3 id="function-test_class">
Function <code>thrust::test&#95;class::&gt;::test&#95;class</code>
</h3>

<code class="doxybook">
<span><b>test_class</b>() = default;</span></code>
Construct an empty test class. 

<h3 id="function-test_class">
Function <code>thrust::test&#95;class::&gt;::test&#95;class</code>
</h3>

<code class="doxybook">
<span>__host__ constexpr __device__ </span><span><b>test_class</b>(int);</span></code>
Construct a test class. 

<h3 id="function-test_member_function">
Function <code>thrust::test&#95;class::&gt;::test&#95;member&#95;function</code>
</h3>

<code class="doxybook">
<span>virtual __host__ constexpr virtual __device__ int </span><span><b>test_member_function</b>() = 0;</span></code>
<code>test&#95;member&#95;function</code> is a function intended to exercise and test Doxybook rendering. 

<h3 id="function-test_parameter_overflow_member_function">
Function <code>thrust::test&#95;class::&gt;::test&#95;parameter&#95;overflow&#95;member&#95;function</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T = test&#95;predefined&#95;friend&#95;struct&lt;int, int, int, int, int, int, int, int, int, int, int, int&gt;,</span>
<span>&nbsp;&nbsp;typename U = test&#95;predefined&#95;friend&#95;struct&lt;int, int, int, int, int, int, int, int, int, int, int, int&gt;,</span>
<span>&nbsp;&nbsp;typename V = test&#95;predefined&#95;friend&#95;struct&lt;int, int, int, int, int, int, int, int, int, int, int, int&gt;&gt;</span>
<span><a href="/api/classes/structthrust_1_1test__predefined__friend__struct.html">test_predefined_friend_struct</a>< int, int, int, int, int, int, int, int, int, int, int, int, int, int, int > </span><span><b>test_parameter_overflow_member_function</b>(<a href="/api/classes/structthrust_1_1test__predefined__friend__struct.html">test_predefined_friend_struct</a>< int, int, int, int, int, int, int, int, int, int, int, int, int, int, int > t,</span>
<span>&nbsp;&nbsp;<a href="/api/classes/structthrust_1_1test__predefined__friend__struct.html">test_predefined_friend_struct</a>< int, int, int, int, int, int, int, int, int, int, int, int, int, int, int > u,</span>
<span>&nbsp;&nbsp;<a href="/api/classes/structthrust_1_1test__predefined__friend__struct.html">test_predefined_friend_struct</a>< int, int, int, int, int, int, int, int, int, int, int, int, int, int, int > v);</span></code>
<code>test&#95;parameter&#95;overflow&#95;member&#95;function</code> is a function intended to test Doxybook's rendering of function and template parameters that exceed the length of a line. 


## Protected Member Functions

<h3 id="function-test_protected_member_function">
Function <code>thrust::test&#95;class::&gt;::test&#95;protected&#95;member&#95;function</code>
</h3>

<code class="doxybook">
<span>__device__ auto </span><span><b>test_protected_member_function</b>();</span></code>
<code>test&#95;protected&#95;member&#95;function</code> is a function intended to exercise and test Doxybook rendering. 


