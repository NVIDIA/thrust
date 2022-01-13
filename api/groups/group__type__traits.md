---
title: Type Traits
parent: Utility
grand_parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Type Traits

<code class="doxybook">
<span class="doxybook-comment">/* Customization point that can be customized to indicate that an iterator type <code>Iterator</code> satisfies <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>, aka it points to elements that are contiguous in memory.  */</span><span>template &lt;typename Iterator&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1proclaim__contiguous__iterator.html">thrust::proclaim&#95;contiguous&#95;iterator</a></b>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/types/integral_constant"><code>std::integral&#95;constant</code></a> whose value is <code>(... && Bs)</code>.  */</span><span>template &lt;bool... Bs&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1conjunction__value.html">thrust::conjunction&#95;value</a></b>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/types/integral_constant"><code>std::integral&#95;constant</code></a> whose value is <code>(... || Bs)</code>.  */</span><span>template &lt;bool... Bs&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1disjunction__value.html">thrust::disjunction&#95;value</a></b>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/types/integral_constant"><code>std::integral&#95;constant</code></a> whose value is <code>!Bs</code>.  */</span><span>template &lt;bool B&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1negation__value.html">thrust::negation&#95;value</a></b>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that removes <a href="https://en.cppreference.com/w/cpp/language/cv">const-volatile qualifiers</a> and <a href="https://en.cppreference.com/w/cpp/language/reference">references</a> from <code>T</code>. Equivalent to <code>remove&#95;cv&#95;t&lt;remove&#95;reference&#95;t&lt;T&gt;&gt;</code>.  */</span><span>template &lt;typename T&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1remove__cvref.html">thrust::remove&#95;cvref</a></b>;</span>
<br>
<span>template &lt;typename...&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1voider.html">thrust::voider</a></b>;</span>
<br>
<span class="doxybook-comment">/* A compile-time sequence of <a href="https://en.cppreference.com/w/cpp/language/constant_expression#Integral_constant_expression">_integral constants_</a> of type <code>T</code> with values <code>Is...</code>.  */</span><span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;T... Is&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#using-integer-sequence">thrust::integer&#95;sequence</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* A compile-time sequence of type <a href="https://en.cppreference.com/w/cpp/types/size_t">std::size_t</a> with values <code>Is...</code>.  */</span><span>template &lt;std::size_t... Is&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#using-index-sequence">thrust::index&#95;sequence</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* Create a new <code>integer&#95;sequence</code> with elements <code>0, 1, 2, ..., N - 1</code> of type <code>T</code>.  */</span><span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;std::size_t N&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#using-make-integer-sequence">thrust::make&#95;integer&#95;sequence</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* Create a new <code>integer&#95;sequence</code> with elements <code>0, 1, 2, ..., N - 1</code> of type <a href="https://en.cppreference.com/w/cpp/types/size_t">std::size_t</a>.  */</span><span>template &lt;std::size_t N&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#using-make-index-sequence">thrust::make&#95;index&#95;sequence</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* Create a new <code>integer&#95;sequence</code> with elements <code>N - 1, N - 2, N - 3, ..., 0</code>.  */</span><span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;std::size_t N&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#using-make-reversed-integer-sequence">thrust::make&#95;reversed&#95;integer&#95;sequence</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* Create a new <code>index&#95;sequence</code> with elements <code>N - 1, N - 2, N - 3, ..., 0</code>.  */</span><span>template &lt;std::size_t N&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#using-make-reversed-index-sequence">thrust::make&#95;reversed&#95;index&#95;sequence</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* Add a new element to the front of an <code>integer&#95;sequence</code>.  */</span><span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;T Value,</span>
<span>&nbsp;&nbsp;typename Sequence&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#using-integer-sequence-push-front">thrust::integer&#95;sequence&#95;push&#95;front</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* Add a new element to the back of an <code>integer&#95;sequence</code>.  */</span><span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;T Value,</span>
<span>&nbsp;&nbsp;typename Sequence&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#using-integer-sequence-push-back">thrust::integer&#95;sequence&#95;push&#95;back</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>Iterator</code> satisfies <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>, aka it points to elements that are contiguous in memory, and <code>false&#95;type</code> otherwise.  */</span><span>template &lt;typename Iterator&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#using-is-contiguous-iterator">thrust::is&#95;contiguous&#95;iterator</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>T</code> is an _ExecutionPolicy_ and <code>false&#95;type</code> otherwise.  */</span><span>template &lt;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#using-is-execution-policy">thrust::is&#95;execution&#95;policy</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a> equivalent to <code>operator&lt;</code>, and <code>false&#95;type</code> otherwise.  */</span><span>template &lt;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#using-is-operator-less-function-object">thrust::is&#95;operator&#95;less&#95;function&#95;object</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a> equivalent to <code>operator&gt;</code>, and <code>false&#95;type</code> otherwise.  */</span><span>template &lt;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#using-is-operator-greater-function-object">thrust::is&#95;operator&#95;greater&#95;function&#95;object</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a> equivalent to <code>operator&lt;</code> or <code>operator&gt;</code>, and <code>false&#95;type</code> otherwise.  */</span><span>template &lt;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#using-is-operator-less-or-greater-function-object">thrust::is&#95;operator&#95;less&#95;or&#95;greater&#95;function&#95;object</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/FunctionObject">FunctionObject</a> equivalent to <code>operator+</code>, and <code>false&#95;type</code> otherwise.  */</span><span>template &lt;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#using-is-operator-plus-function-object">thrust::is&#95;operator&#95;plus&#95;function&#95;object</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/types/integral_constant"><code>std::integral&#95;constant</code></a> whose value is <code>(... && Ts::value)</code>.  */</span><span>template &lt;typename... Ts&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#using-conjunction">thrust::conjunction</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/types/integral_constant"><code>std::integral&#95;constant</code></a> whose value is <code>(... || Ts::value)</code>.  */</span><span>template &lt;typename... Ts&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#using-disjunction">thrust::disjunction</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/types/integral_constant"><code>std::integral&#95;constant</code></a> whose value is <code>!Ts::value</code>.  */</span><span>template &lt;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#using-negation">thrust::negation</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* Type alias that removes <a href="https://en.cppreference.com/w/cpp/language/cv">const-volatile qualifiers</a> and <a href="https://en.cppreference.com/w/cpp/language/reference">references</a> from <code>T</code>. Equivalent to <code>remove&#95;cv&#95;t&lt;remove&#95;reference&#95;t&lt;T&gt;&gt;</code>.  */</span><span>template &lt;typename T&gt;</span>
<span>using <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#using-remove-cvref-t">thrust::remove&#95;cvref&#95;t</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> that is <code>true</code> if <code>Iterator</code> satisfies <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>, aka it points to elements that are contiguous in memory, and <code>false</code> otherwise.  */</span><span>template &lt;typename Iterator&gt;</span>
<span>constexpr bool <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#variable-is-contiguous-iterator-v">thrust::is&#95;contiguous&#95;iterator&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> that is <code>true</code> if <code>T</code> is an _ExecutionPolicy_ and <code>false</code> otherwise.  */</span><span>template &lt;typename T&gt;</span>
<span>constexpr bool <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#variable-is-execution-policy-v">thrust::is&#95;execution&#95;policy&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> that is <code>true</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a> equivalent to <code>operator&lt;</code>, and <code>false</code> otherwise.  */</span><span>template &lt;typename T&gt;</span>
<span>constexpr bool <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#variable-is-operator-less-function-object-v">thrust::is&#95;operator&#95;less&#95;function&#95;object&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> that is <code>true</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a> equivalent to <code>operator&gt;</code>, and <code>false</code> otherwise.  */</span><span>template &lt;typename T&gt;</span>
<span>constexpr bool <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#variable-is-operator-greater-function-object-v">thrust::is&#95;operator&#95;greater&#95;function&#95;object&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> that is <code>true</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a> equivalent to <code>operator&lt;</code> or <code>operator&gt;</code>, and <code>false</code> otherwise.  */</span><span>template &lt;typename T&gt;</span>
<span>constexpr bool <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#variable-is-operator-less-or-greater-function-object-v">thrust::is&#95;operator&#95;less&#95;or&#95;greater&#95;function&#95;object&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> that is <code>true</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/FunctionObject">FunctionObject</a> equivalent to <code>operator&lt;</code>, and <code>false</code> otherwise.  */</span><span>template &lt;typename T&gt;</span>
<span>constexpr bool <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#variable-is-operator-plus-function-object-v">thrust::is&#95;operator&#95;plus&#95;function&#95;object&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> whose value is <code>(... && Ts::value)</code>.  */</span><span>template &lt;typename... Ts&gt;</span>
<span>constexpr bool <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#variable-conjunction-v">thrust::conjunction&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> whose value is <code>(... || Ts::value)</code>.  */</span><span>template &lt;typename... Ts&gt;</span>
<span>constexpr bool <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#variable-disjunction-v">thrust::disjunction&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> whose value is <code>!Ts::value</code>.  */</span><span>template &lt;typename T&gt;</span>
<span>constexpr bool <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#variable-negation-v">thrust::negation&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> whose value is <code>(... && Bs)</code>.  */</span><span>template &lt;bool... Bs&gt;</span>
<span>constexpr bool <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#variable-conjunction-value-v">thrust::conjunction&#95;value&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> whose value is <code>(... || Bs)</code>.  */</span><span>template &lt;bool... Bs&gt;</span>
<span>constexpr bool <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#variable-disjunction-value-v">thrust::disjunction&#95;value&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> whose value is <code>!Ts::value</code>.  */</span><span>template &lt;bool B&gt;</span>
<span>constexpr bool <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#variable-negation-value-v">thrust::negation&#95;value&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span>#define <b><a href="{{ site.baseurl }}/api/groups/group__type__traits.html#define-thrust-proclaim-contiguous-iterator">THRUST&#95;PROCLAIM&#95;CONTIGUOUS&#95;ITERATOR</a></b> = <i>see below</i>;</span>
</code>

## Member Classes

<h3 id="struct-thrustproclaim-contiguous-iterator">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1proclaim__contiguous__iterator.html">Struct <code>thrust::proclaim&#95;contiguous&#95;iterator</code>
</a>
</h3>

Customization point that can be customized to indicate that an iterator type <code>Iterator</code> satisfies <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>, aka it points to elements that are contiguous in memory. 

**Inherits From**:
`false_type`

<h3 id="struct-thrustconjunction-value">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1conjunction__value.html">Struct <code>thrust::conjunction&#95;value</code>
</a>
</h3>

<a href="https://en.cppreference.com/w/cpp/types/integral_constant"><code>std::integral&#95;constant</code></a> whose value is <code>(... && Bs)</code>. 

<h3 id="struct-thrustdisjunction-value">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1disjunction__value.html">Struct <code>thrust::disjunction&#95;value</code>
</a>
</h3>

<a href="https://en.cppreference.com/w/cpp/types/integral_constant"><code>std::integral&#95;constant</code></a> whose value is <code>(... || Bs)</code>. 

<h3 id="struct-thrustnegation-value">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1negation__value.html">Struct <code>thrust::negation&#95;value</code>
</a>
</h3>

<a href="https://en.cppreference.com/w/cpp/types/integral_constant"><code>std::integral&#95;constant</code></a> whose value is <code>!Bs</code>. 

<h3 id="struct-thrustremove-cvref">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1remove__cvref.html">Struct <code>thrust::remove&#95;cvref</code>
</a>
</h3>

<a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that removes <a href="https://en.cppreference.com/w/cpp/language/cv">const-volatile qualifiers</a> and <a href="https://en.cppreference.com/w/cpp/language/reference">references</a> from <code>T</code>. Equivalent to <code>remove&#95;cv&#95;t&lt;remove&#95;reference&#95;t&lt;T&gt;&gt;</code>. 

<h3 id="struct-thrustvoider">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1voider.html">Struct <code>thrust::voider</code>
</a>
</h3>


## Types

<h3 id="using-integer-sequence">
Type Alias <code>thrust::integer&#95;sequence</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;T... Is&gt;</span>
<span>using <b>integer_sequence</b> = std::integer&#95;sequence&lt; T, Is... &gt;;</span></code>
A compile-time sequence of <a href="https://en.cppreference.com/w/cpp/language/constant_expression#Integral_constant_expression">_integral constants_</a> of type <code>T</code> with values <code>Is...</code>. 

**See**:
* <a href="https://en.cppreference.com/w/cpp/language/constant_expression#Integral_constant_expression">_integral constants_</a>
* index_sequence 
* make_integer_sequence 
* make_reversed_integer_sequence 
* make_index_sequence 
* make_reversed_index_sequence 
* integer_sequence_push_front 
* integer_sequence_push_back 
* <a href="https://en.cppreference.com/w/cpp/utility/integer_sequence"><code>std::integer&#95;sequence</code></a>

<h3 id="using-index-sequence">
Type Alias <code>thrust::index&#95;sequence</code>
</h3>

<code class="doxybook">
<span>template &lt;std::size_t... Is&gt;</span>
<span>using <b>index_sequence</b> = std::index&#95;sequence&lt; Is... &gt;;</span></code>
A compile-time sequence of type <a href="https://en.cppreference.com/w/cpp/types/size_t">std::size_t</a> with values <code>Is...</code>. 

**See**:
* integer_sequence 
* make_integer_sequence 
* make_reversed_integer_sequence 
* make_index_sequence 
* make_reversed_index_sequence 
* integer_sequence_push_front 
* integer_sequence_push_back 
* <a href="https://en.cppreference.com/w/cpp/utility/integer_sequence"><code>std::index&#95;sequence</code></a>

<h3 id="using-make-integer-sequence">
Type Alias <code>thrust::make&#95;integer&#95;sequence</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;std::size_t N&gt;</span>
<span>using <b>make_integer_sequence</b> = std::make&#95;integer&#95;sequence&lt; T, N &gt;;</span></code>
Create a new <code>integer&#95;sequence</code> with elements <code>0, 1, 2, ..., N - 1</code> of type <code>T</code>. 

**See**:
* integer_sequence 
* index_sequence 
* make_reversed_integer_sequence 
* make_index_sequence 
* make_reversed_index_sequence 
* <a href="https://en.cppreference.com/w/cpp/utility/integer_sequence"><code>std::make&#95;integer&#95;sequence</code></a>

<h3 id="using-make-index-sequence">
Type Alias <code>thrust::make&#95;index&#95;sequence</code>
</h3>

<code class="doxybook">
<span>template &lt;std::size_t N&gt;</span>
<span>using <b>make_index_sequence</b> = std::make&#95;index&#95;sequence&lt; N &gt;;</span></code>
Create a new <code>integer&#95;sequence</code> with elements <code>0, 1, 2, ..., N - 1</code> of type <a href="https://en.cppreference.com/w/cpp/types/size_t">std::size_t</a>. 

**See**:
* integer_sequence 
* index_sequence 
* make_integer_sequence 
* make_reversed_integer_sequence 
* make_reversed_index_sequence 
* <a href="https://en.cppreference.com/w/cpp/utility/integer_sequence"><code>std::make&#95;index&#95;sequence</code></a>

<h3 id="using-make-reversed-integer-sequence">
Type Alias <code>thrust::make&#95;reversed&#95;integer&#95;sequence</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;std::size_t N&gt;</span>
<span>using <b>make_reversed_integer_sequence</b> = typename detail::make&#95;reversed&#95;integer&#95;sequence&#95;impl&lt; T, N &gt;::type;</span></code>
Create a new <code>integer&#95;sequence</code> with elements <code>N - 1, N - 2, N - 3, ..., 0</code>. 

**See**:
* integer_sequence 
* index_sequence 
* make_integer_sequence 
* make_index_sequence 
* make_reversed_index_sequence 

<h3 id="using-make-reversed-index-sequence">
Type Alias <code>thrust::make&#95;reversed&#95;index&#95;sequence</code>
</h3>

<code class="doxybook">
<span>template &lt;std::size_t N&gt;</span>
<span>using <b>make_reversed_index_sequence</b> = make&#95;reversed&#95;integer&#95;sequence&lt; std::size&#95;t, N &gt;;</span></code>
Create a new <code>index&#95;sequence</code> with elements <code>N - 1, N - 2, N - 3, ..., 0</code>. 

**See**:
* integer_sequence 
* index_sequence 
* make_integer_sequence 
* make_reversed_integer_sequence 
* make_reversed_index_sequence 

<h3 id="using-integer-sequence-push-front">
Type Alias <code>thrust::integer&#95;sequence&#95;push&#95;front</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;T Value,</span>
<span>&nbsp;&nbsp;typename Sequence&gt;</span>
<span>using <b>integer_sequence_push_front</b> = typename detail::integer&#95;sequence&#95;push&#95;front&#95;impl&lt; T, Value, Sequence &gt;::type;</span></code>
Add a new element to the front of an <code>integer&#95;sequence</code>. 

**See**:
* integer_sequence 
* index_sequence 
* make_integer_sequence 
* make_index_sequence 

<h3 id="using-integer-sequence-push-back">
Type Alias <code>thrust::integer&#95;sequence&#95;push&#95;back</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;T Value,</span>
<span>&nbsp;&nbsp;typename Sequence&gt;</span>
<span>using <b>integer_sequence_push_back</b> = typename detail::integer&#95;sequence&#95;push&#95;back&#95;impl&lt; T, Value, Sequence &gt;::type;</span></code>
Add a new element to the back of an <code>integer&#95;sequence</code>. 

**See**:
* integer_sequence 
* index_sequence 
* make_integer_sequence 
* make_index_sequence 

<h3 id="using-is-contiguous-iterator">
Type Alias <code>thrust::is&#95;contiguous&#95;iterator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Iterator&gt;</span>
<span>using <b>is_contiguous_iterator</b> = detail::is&#95;contiguous&#95;iterator&#95;impl&lt; Iterator &gt;;</span></code>
<a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>Iterator</code> satisfies <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>, aka it points to elements that are contiguous in memory, and <code>false&#95;type</code> otherwise. 

**See**:
* is_contiguous_iterator_v 
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1proclaim__contiguous__iterator.html">proclaim_contiguous_iterator</a>
* <a href="{{ site.baseurl }}/api/groups/group__type__traits.html#define-thrust-proclaim-contiguous-iterator">THRUST_PROCLAIM_CONTIGUOUS_ITERATOR</a>

<h3 id="using-is-execution-policy">
Type Alias <code>thrust::is&#95;execution&#95;policy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>is_execution_policy</b> = detail::is&#95;base&#95;of&lt; detail::execution&#95;policy&#95;marker, T &gt;;</span></code>
<a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>T</code> is an _ExecutionPolicy_ and <code>false&#95;type</code> otherwise. 

<h3 id="using-is-operator-less-function-object">
Type Alias <code>thrust::is&#95;operator&#95;less&#95;function&#95;object</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>is_operator_less_function_object</b> = detail::is&#95;operator&#95;less&#95;function&#95;object&#95;impl&lt; T &gt;;</span></code>
<a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a> equivalent to <code>operator&lt;</code>, and <code>false&#95;type</code> otherwise. 

**See**:
* is_operator_less_function_object_v 
* is_operator_greater_function_object 
* is_operator_less_or_greater_function_object 
* is_operator_plus_function_object 

<h3 id="using-is-operator-greater-function-object">
Type Alias <code>thrust::is&#95;operator&#95;greater&#95;function&#95;object</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>is_operator_greater_function_object</b> = detail::is&#95;operator&#95;greater&#95;function&#95;object&#95;impl&lt; T &gt;;</span></code>
<a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a> equivalent to <code>operator&gt;</code>, and <code>false&#95;type</code> otherwise. 

**See**:
* is_operator_greater_function_object_v 
* is_operator_less_function_object 
* is_operator_less_or_greater_function_object 
* is_operator_plus_function_object 

<h3 id="using-is-operator-less-or-greater-function-object">
Type Alias <code>thrust::is&#95;operator&#95;less&#95;or&#95;greater&#95;function&#95;object</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>is_operator_less_or_greater_function_object</b> = integral&#95;constant&lt; bool, detail::is&#95;operator&#95;less&#95;function&#95;object&#95;impl&lt; T &gt;::value||detail::is&#95;operator&#95;greater&#95;function&#95;object&#95;impl&lt; T &gt;::value &gt;;</span></code>
<a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a> equivalent to <code>operator&lt;</code> or <code>operator&gt;</code>, and <code>false&#95;type</code> otherwise. 

**See**:
* is_operator_less_or_greater_function_object_v 
* is_operator_less_function_object 
* is_operator_greater_function_object 
* is_operator_plus_function_object 

<h3 id="using-is-operator-plus-function-object">
Type Alias <code>thrust::is&#95;operator&#95;plus&#95;function&#95;object</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>is_operator_plus_function_object</b> = detail::is&#95;operator&#95;plus&#95;function&#95;object&#95;impl&lt; T &gt;;</span></code>
<a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/FunctionObject">FunctionObject</a> equivalent to <code>operator+</code>, and <code>false&#95;type</code> otherwise. 

**See**:
* is_operator_plus_function_object_v 
* is_operator_less_function_object 
* is_operator_greater_function_object 
* is_operator_less_or_greater_function_object 

<h3 id="using-conjunction">
Type Alias <code>thrust::conjunction</code>
</h3>

<code class="doxybook">
<span>template &lt;typename... Ts&gt;</span>
<span>using <b>conjunction</b> = std::conjunction&lt; Ts... &gt;;</span></code>
<a href="https://en.cppreference.com/w/cpp/types/integral_constant"><code>std::integral&#95;constant</code></a> whose value is <code>(... && Ts::value)</code>. 

**See**:
* conjunction_v 
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1conjunction__value.html">conjunction_value</a>
* <a href="https://en.cppreference.com/w/cpp/types/conjunction"><code>std::conjunction</code></a>

<h3 id="using-disjunction">
Type Alias <code>thrust::disjunction</code>
</h3>

<code class="doxybook">
<span>template &lt;typename... Ts&gt;</span>
<span>using <b>disjunction</b> = std::disjunction&lt; Ts... &gt;;</span></code>
<a href="https://en.cppreference.com/w/cpp/types/integral_constant"><code>std::integral&#95;constant</code></a> whose value is <code>(... || Ts::value)</code>. 

**See**:
* disjunction_v 
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1disjunction__value.html">disjunction_value</a>
* <a href="https://en.cppreference.com/w/cpp/types/disjunction"><code>std::disjunction</code></a>

<h3 id="using-negation">
Type Alias <code>thrust::negation</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>negation</b> = std::negation&lt; T &gt;;</span></code>
<a href="https://en.cppreference.com/w/cpp/types/integral_constant"><code>std::integral&#95;constant</code></a> whose value is <code>!Ts::value</code>. 

**See**:
* negation_v 
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1negation__value.html">negation_value</a>
* <a href="https://en.cppreference.com/w/cpp/types/negation"><code>std::negation</code></a>

<h3 id="using-remove-cvref-t">
Type Alias <code>thrust::remove&#95;cvref&#95;t</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>remove_cvref_t</b> = typename remove&#95;cvref&lt; T &gt;::type;</span></code>
Type alias that removes <a href="https://en.cppreference.com/w/cpp/language/cv">const-volatile qualifiers</a> and <a href="https://en.cppreference.com/w/cpp/language/reference">references</a> from <code>T</code>. Equivalent to <code>remove&#95;cv&#95;t&lt;remove&#95;reference&#95;t&lt;T&gt;&gt;</code>. 

**See**:
* <a href="https://en.cppreference.com/w/cpp/types/remove_cvref">std::remove_cvref</a>
* <a href="https://en.cppreference.com/w/cpp/types/remove_cv">std::remove_cv</a>
* <a href="https://en.cppreference.com/w/cpp/types/remove_cv">std::remove_const</a>
* <a href="https://en.cppreference.com/w/cpp/types/remove_cv">std::remove_volatile</a>
* <a href="https://en.cppreference.com/w/cpp/types/remove_cv">std::remove_reference</a>


## Variables

<h3 id="variable-is-contiguous-iterator-v">
Variable <code>thrust::is&#95;contiguous&#95;iterator&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Iterator&gt;</span>
<span>constexpr bool <b>is_contiguous_iterator_v</b> = is&#95;contiguous&#95;iterator&lt;Iterator&gt;::value;</span></code>
<code>constexpr bool</code> that is <code>true</code> if <code>Iterator</code> satisfies <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>, aka it points to elements that are contiguous in memory, and <code>false</code> otherwise. 

**See**:
* is_contiguous_iterator 
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1proclaim__contiguous__iterator.html">proclaim_contiguous_iterator</a>
* <a href="{{ site.baseurl }}/api/groups/group__type__traits.html#define-thrust-proclaim-contiguous-iterator">THRUST_PROCLAIM_CONTIGUOUS_ITERATOR</a>

<h3 id="variable-is-execution-policy-v">
Variable <code>thrust::is&#95;execution&#95;policy&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>constexpr bool <b>is_execution_policy_v</b> = is&#95;execution&#95;policy&lt;T&gt;::value;</span></code>
<code>constexpr bool</code> that is <code>true</code> if <code>T</code> is an _ExecutionPolicy_ and <code>false</code> otherwise. 

<h3 id="variable-is-operator-less-function-object-v">
Variable <code>thrust::is&#95;operator&#95;less&#95;function&#95;object&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>constexpr bool <b>is_operator_less_function_object_v</b> = is&#95;operator&#95;less&#95;function&#95;object&lt;T&gt;::value;</span></code>
<code>constexpr bool</code> that is <code>true</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a> equivalent to <code>operator&lt;</code>, and <code>false</code> otherwise. 

**See**:
* is_operator_less_function_object 
* is_operator_greater_function_object 
* is_operator_less_or_greater_function_object 
* is_operator_plus_function_object 

<h3 id="variable-is-operator-greater-function-object-v">
Variable <code>thrust::is&#95;operator&#95;greater&#95;function&#95;object&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>constexpr bool <b>is_operator_greater_function_object_v</b> = is&#95;operator&#95;greater&#95;function&#95;object&lt;T&gt;::value;</span></code>
<code>constexpr bool</code> that is <code>true</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a> equivalent to <code>operator&gt;</code>, and <code>false</code> otherwise. 

**See**:
* is_operator_greater_function_object 
* is_operator_less_function_object 
* is_operator_less_or_greater_function_object 
* is_operator_plus_function_object 

<h3 id="variable-is-operator-less-or-greater-function-object-v">
Variable <code>thrust::is&#95;operator&#95;less&#95;or&#95;greater&#95;function&#95;object&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>constexpr bool <b>is_operator_less_or_greater_function_object_v</b> = is&#95;operator&#95;less&#95;or&#95;greater&#95;function&#95;object&lt;T&gt;::value;</span></code>
<code>constexpr bool</code> that is <code>true</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a> equivalent to <code>operator&lt;</code> or <code>operator&gt;</code>, and <code>false</code> otherwise. 

**See**:
* is_operator_less_or_greater_function_object 
* is_operator_less_function_object 
* is_operator_greater_function_object 
* is_operator_plus_function_object 

<h3 id="variable-is-operator-plus-function-object-v">
Variable <code>thrust::is&#95;operator&#95;plus&#95;function&#95;object&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>constexpr bool <b>is_operator_plus_function_object_v</b> = is&#95;operator&#95;plus&#95;function&#95;object&lt;T&gt;::value;</span></code>
<code>constexpr bool</code> that is <code>true</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/FunctionObject">FunctionObject</a> equivalent to <code>operator&lt;</code>, and <code>false</code> otherwise. 

**See**:
* is_operator_plus_function_object 
* is_operator_less_function_object 
* is_operator_greater_function_object 
* is_operator_less_or_greater_function_object 

<h3 id="variable-conjunction-v">
Variable <code>thrust::conjunction&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;typename... Ts&gt;</span>
<span>constexpr bool <b>conjunction_v</b> = conjunction&lt;Ts...&gt;::value;</span></code>
<code>constexpr bool</code> whose value is <code>(... && Ts::value)</code>. 

**See**:
* conjunction 
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1conjunction__value.html">conjunction_value</a>
* <a href="https://en.cppreference.com/w/cpp/types/conjunction"><code>std::conjunction</code></a>

<h3 id="variable-disjunction-v">
Variable <code>thrust::disjunction&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;typename... Ts&gt;</span>
<span>constexpr bool <b>disjunction_v</b> = disjunction&lt;Ts...&gt;::value;</span></code>
<code>constexpr bool</code> whose value is <code>(... || Ts::value)</code>. 

**See**:
* disjunction 
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1disjunction__value.html">disjunction_value</a>
* <a href="https://en.cppreference.com/w/cpp/types/disjunction"><code>std::disjunction</code></a>

<h3 id="variable-negation-v">
Variable <code>thrust::negation&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>constexpr bool <b>negation_v</b> = negation&lt;T&gt;::value;</span></code>
<code>constexpr bool</code> whose value is <code>!Ts::value</code>. 

**See**:
* negation 
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1negation__value.html">negation_value</a>
* <a href="https://en.cppreference.com/w/cpp/types/negation"><code>std::negation</code></a>

<h3 id="variable-conjunction-value-v">
Variable <code>thrust::conjunction&#95;value&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;bool... Bs&gt;</span>
<span>constexpr bool <b>conjunction_value_v</b> = conjunction&#95;value&lt;Bs...&gt;::value;</span></code>
<code>constexpr bool</code> whose value is <code>(... && Bs)</code>. 

**See**:
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1conjunction__value.html">conjunction_value</a>
* conjunction 
* <a href="https://en.cppreference.com/w/cpp/types/conjunction"><code>std::conjunction</code></a>

<h3 id="variable-disjunction-value-v">
Variable <code>thrust::disjunction&#95;value&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;bool... Bs&gt;</span>
<span>constexpr bool <b>disjunction_value_v</b> = disjunction&#95;value&lt;Bs...&gt;::value;</span></code>
<code>constexpr bool</code> whose value is <code>(... || Bs)</code>. 

**See**:
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1disjunction__value.html">disjunction_value</a>
* disjunction 
* <a href="https://en.cppreference.com/w/cpp/types/disjunction"><code>std::disjunction</code></a>

<h3 id="variable-negation-value-v">
Variable <code>thrust::negation&#95;value&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;bool B&gt;</span>
<span>constexpr bool <b>negation_value_v</b> = negation&#95;value&lt;B&gt;::value;</span></code>
<code>constexpr bool</code> whose value is <code>!Ts::value</code>. 

**See**:
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1negation__value.html">negation_value</a>
* negation 
* <a href="https://en.cppreference.com/w/cpp/types/negation"><code>std::negation</code></a>


## Macros

<h3 id="define-thrust-proclaim-contiguous-iterator">
Define <code>THRUST&#95;PROCLAIM&#95;CONTIGUOUS&#95;ITERATOR</code>
</h3>

Declares that the iterator <code>Iterator</code> is <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a> by specializing <code>proclaim&#95;contiguous&#95;iterator</code>. 

**See**:
* is_contiguous_iterator 
* proclaim_contiguous_iterator 


