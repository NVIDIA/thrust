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
<span>struct <b><a href="/api/classes/structproclaim__contiguous__iterator.html">proclaim&#95;contiguous&#95;iterator</a></b>;</span>
<br>
<span class="doxybook-comment">/* <a href="http://eel.is/c++draft/namespace.std#def:customization_point">_customization point_</a> that can be specialized customized to indicate that a type <code>T</code> is <a href="https://wg21.link/P1144">_TriviallyRelocatable_</a>, aka it can be bitwise copied with a facility like <a href="https://en.cppreference.com/w/cpp/string/byte/memcpy"><code>std::memcpy</code></a>.  */</span><span>template &lt;typename T&gt;</span>
<span>struct <b><a href="/api/classes/structproclaim__trivially__relocatable.html">proclaim&#95;trivially&#95;relocatable</a></b>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/types/integral_constant"><code>std::integral&#95;constant</code></a> whose value is <code>(... && Bs)</code>.  */</span><span>template &lt;bool... Bs&gt;</span>
<span>struct <b><a href="/api/classes/structconjunction__value.html">conjunction&#95;value</a></b>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/types/integral_constant"><code>std::integral&#95;constant</code></a> whose value is <code>(... || Bs)</code>.  */</span><span>template &lt;bool... Bs&gt;</span>
<span>struct <b><a href="/api/classes/structdisjunction__value.html">disjunction&#95;value</a></b>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/types/integral_constant"><code>std::integral&#95;constant</code></a> whose value is <code>!Bs</code>.  */</span><span>template &lt;bool B&gt;</span>
<span>struct <b><a href="/api/classes/structnegation__value.html">negation&#95;value</a></b>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that removes <a href="https://en.cppreference.com/w/cpp/language/cv">const-volatile qualifiers</a> and <a href="https://en.cppreference.com/w/cpp/language/reference">references</a> from <code>T</code>. Equivalent to <code>remove&#95;cv&#95;t&lt;remove&#95;reference&#95;t&lt;T&gt;&gt;</code>.  */</span><span>template &lt;typename T&gt;</span>
<span>struct <b><a href="/api/classes/structremove__cvref.html">remove&#95;cvref</a></b>;</span>
<br>
<span>template &lt;typename...&gt;</span>
<span>struct <b><a href="/api/classes/structvoider.html">voider</a></b>;</span>
<br>
<span class="doxybook-comment">/* A compile-time sequence of <a href="https://en.cppreference.com/w/cpp/language/constant_expression#Integral_constant_expression">_integral constants_</a> of type <code>T</code> with values <code>Is...</code>.  */</span><span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;T... Is&gt;</span>
<span>using <b><a href="/api/groups/group__type__traits.html#using-integer_sequence">integer&#95;sequence</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* A compile-time sequence of type <a href="https://en.cppreference.com/w/cpp/types/size_t">std::size_t</a> with values <code>Is...</code>.  */</span><span>template &lt;std::size_t... Is&gt;</span>
<span>using <b><a href="/api/groups/group__type__traits.html#using-index_sequence">index&#95;sequence</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* Create a new <code>integer&#95;sequence</code> with elements <code>0, 1, 2, ..., N - 1</code> of type <code>T</code>.  */</span><span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;std::size_t N&gt;</span>
<span>using <b><a href="/api/groups/group__type__traits.html#using-make_integer_sequence">make&#95;integer&#95;sequence</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* Create a new <code>integer&#95;sequence</code> with elements <code>0, 1, 2, ..., N - 1</code> of type <a href="https://en.cppreference.com/w/cpp/types/size_t">std::size_t</a>.  */</span><span>template &lt;std::size_t N&gt;</span>
<span>using <b><a href="/api/groups/group__type__traits.html#using-make_index_sequence">make&#95;index&#95;sequence</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* Create a new <code>integer&#95;sequence</code> with elements <code>N - 1, N - 2, N - 3, ..., 0</code>.  */</span><span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;std::size_t N&gt;</span>
<span>using <b><a href="/api/groups/group__type__traits.html#using-make_reversed_integer_sequence">make&#95;reversed&#95;integer&#95;sequence</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* Create a new <code>index&#95;sequence</code> with elements <code>N - 1, N - 2, N - 3, ..., 0</code>.  */</span><span>template &lt;std::size_t N&gt;</span>
<span>using <b><a href="/api/groups/group__type__traits.html#using-make_reversed_index_sequence">make&#95;reversed&#95;index&#95;sequence</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* Add a new element to the front of an <code>integer&#95;sequence</code>.  */</span><span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;T Value,</span>
<span>&nbsp;&nbsp;typename Sequence&gt;</span>
<span>using <b><a href="/api/groups/group__type__traits.html#using-integer_sequence_push_front">integer&#95;sequence&#95;push&#95;front</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* Add a new element to the back of an <code>integer&#95;sequence</code>.  */</span><span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;T Value,</span>
<span>&nbsp;&nbsp;typename Sequence&gt;</span>
<span>using <b><a href="/api/groups/group__type__traits.html#using-integer_sequence_push_back">integer&#95;sequence&#95;push&#95;back</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>Iterator</code> satisfies <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>, aka it points to elements that are contiguous in memory, and <code>false&#95;type</code> otherwise.  */</span><span>template &lt;typename Iterator&gt;</span>
<span>using <b><a href="/api/groups/group__type__traits.html#using-is_contiguous_iterator">is&#95;contiguous&#95;iterator</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>T</code> is an _ExecutionPolicy_ and <code>false&#95;type</code> otherwise.  */</span><span>template &lt;typename T&gt;</span>
<span>using <b><a href="/api/groups/group__type__traits.html#using-is_execution_policy">is&#95;execution&#95;policy</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a> equivalent to <code>operator&lt;</code>, and <code>false&#95;type</code> otherwise.  */</span><span>template &lt;typename T&gt;</span>
<span>using <b><a href="/api/groups/group__type__traits.html#using-is_operator_less_function_object">is&#95;operator&#95;less&#95;function&#95;object</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a> equivalent to <code>operator&gt;</code>, and <code>false&#95;type</code> otherwise.  */</span><span>template &lt;typename T&gt;</span>
<span>using <b><a href="/api/groups/group__type__traits.html#using-is_operator_greater_function_object">is&#95;operator&#95;greater&#95;function&#95;object</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a> equivalent to <code>operator&lt;</code> or <code>operator&gt;</code>, and <code>false&#95;type</code> otherwise.  */</span><span>template &lt;typename T&gt;</span>
<span>using <b><a href="/api/groups/group__type__traits.html#using-is_operator_less_or_greater_function_object">is&#95;operator&#95;less&#95;or&#95;greater&#95;function&#95;object</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/FunctionObject">FunctionObject</a> equivalent to <code>operator+</code>, and <code>false&#95;type</code> otherwise.  */</span><span>template &lt;typename T&gt;</span>
<span>using <b><a href="/api/groups/group__type__traits.html#using-is_operator_plus_function_object">is&#95;operator&#95;plus&#95;function&#95;object</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>T</code> is <a href="https://wg21.link/P1144">_TriviallyRelocatable_</a>, aka can be bitwise copied with a facility like <a href="https://en.cppreference.com/w/cpp/string/byte/memcpy"><code>std::memcpy</code></a>, and <code>false&#95;type</code> otherwise.  */</span><span>template &lt;typename T&gt;</span>
<span>using <b><a href="/api/groups/group__type__traits.html#using-is_trivially_relocatable">is&#95;trivially&#95;relocatable</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/named_req/BinaryTypeTrait">_BinaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>From</code> is <a href="https://wg21.link/P1144">_TriviallyRelocatable_</a>, to <code>To</code>, aka can be bitwise copied with a facility like <a href="https://en.cppreference.com/w/cpp/string/byte/memcpy"><code>std::memcpy</code></a>, and <code>false&#95;type</code> otherwise.  */</span><span>template &lt;typename From,</span>
<span>&nbsp;&nbsp;typename To&gt;</span>
<span>using <b><a href="/api/groups/group__type__traits.html#using-is_trivially_relocatable_to">is&#95;trivially&#95;relocatable&#95;to</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/named_req/BinaryTypeTrait">_BinaryTypeTrait_</a> that returns <code>true&#95;type</code> if the element type of <code>FromIterator</code> is <a href="https://wg21.link/P1144">_TriviallyRelocatable_</a>, to the element type of <code>ToIterator</code>, aka can be bitwise copied with a facility like <a href="https://en.cppreference.com/w/cpp/string/byte/memcpy"><code>std::memcpy</code></a>, and <code>false&#95;type</code> otherwise.  */</span><span>template &lt;typename FromIterator,</span>
<span>&nbsp;&nbsp;typename ToIterator&gt;</span>
<span>using <b><a href="/api/groups/group__type__traits.html#using-is_indirectly_trivially_relocatable_to">is&#95;indirectly&#95;trivially&#95;relocatable&#95;to</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/types/integral_constant"><code>std::integral&#95;constant</code></a> whose value is <code>(... && Ts::value)</code>.  */</span><span>template &lt;typename... Ts&gt;</span>
<span>using <b><a href="/api/groups/group__type__traits.html#using-conjunction">conjunction</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/types/integral_constant"><code>std::integral&#95;constant</code></a> whose value is <code>(... || Ts::value)</code>.  */</span><span>template &lt;typename... Ts&gt;</span>
<span>using <b><a href="/api/groups/group__type__traits.html#using-disjunction">disjunction</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <a href="https://en.cppreference.com/w/cpp/types/integral_constant"><code>std::integral&#95;constant</code></a> whose value is <code>!Ts::value</code>.  */</span><span>template &lt;typename T&gt;</span>
<span>using <b><a href="/api/groups/group__type__traits.html#using-negation">negation</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* Type alias that removes <a href="https://en.cppreference.com/w/cpp/language/cv">const-volatile qualifiers</a> and <a href="https://en.cppreference.com/w/cpp/language/reference">references</a> from <code>T</code>. Equivalent to <code>remove&#95;cv&#95;t&lt;remove&#95;reference&#95;t&lt;T&gt;&gt;</code>.  */</span><span>template &lt;typename T&gt;</span>
<span>using <b><a href="/api/groups/group__type__traits.html#using-remove_cvref_t">remove&#95;cvref&#95;t</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> that is <code>true</code> if <code>Iterator</code> satisfies <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>, aka it points to elements that are contiguous in memory, and <code>false</code> otherwise.  */</span><span>template &lt;typename Iterator&gt;</span>
<span>constexpr bool <b><a href="/api/groups/group__type__traits.html#variable-is_contiguous_iterator_v">type&#95;traits::is&#95;contiguous&#95;iterator&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> that is <code>true</code> if <code>T</code> is an _ExecutionPolicy_ and <code>false</code> otherwise.  */</span><span>template &lt;typename T&gt;</span>
<span>constexpr bool <b><a href="/api/groups/group__type__traits.html#variable-is_execution_policy_v">type&#95;traits::is&#95;execution&#95;policy&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> that is <code>true</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a> equivalent to <code>operator&lt;</code>, and <code>false</code> otherwise.  */</span><span>template &lt;typename T&gt;</span>
<span>constexpr bool <b><a href="/api/groups/group__type__traits.html#variable-is_operator_less_function_object_v">type&#95;traits::is&#95;operator&#95;less&#95;function&#95;object&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> that is <code>true</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a> equivalent to <code>operator&gt;</code>, and <code>false</code> otherwise.  */</span><span>template &lt;typename T&gt;</span>
<span>constexpr bool <b><a href="/api/groups/group__type__traits.html#variable-is_operator_greater_function_object_v">type&#95;traits::is&#95;operator&#95;greater&#95;function&#95;object&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> that is <code>true</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a> equivalent to <code>operator&lt;</code> or <code>operator&gt;</code>, and <code>false</code> otherwise.  */</span><span>template &lt;typename T&gt;</span>
<span>constexpr bool <b><a href="/api/groups/group__type__traits.html#variable-is_operator_less_or_greater_function_object_v">type&#95;traits::is&#95;operator&#95;less&#95;or&#95;greater&#95;function&#95;object&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> that is <code>true</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/FunctionObject">FunctionObject</a> equivalent to <code>operator&lt;</code>, and <code>false</code> otherwise.  */</span><span>template &lt;typename T&gt;</span>
<span>constexpr bool <b><a href="/api/groups/group__type__traits.html#variable-is_operator_plus_function_object_v">type&#95;traits::is&#95;operator&#95;plus&#95;function&#95;object&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> that is <code>true</code> if <code>T</code> is <a href="https://wg21.link/P1144">_TriviallyRelocatable_</a>, aka can be bitwise copied with a facility like <a href="https://en.cppreference.com/w/cpp/string/byte/memcpy"><code>std::memcpy</code></a>, and <code>false</code> otherwise.  */</span><span>template &lt;typename T&gt;</span>
<span>constexpr bool <b><a href="/api/groups/group__type__traits.html#variable-is_trivially_relocatable_v">type&#95;traits::is&#95;trivially&#95;relocatable&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> that is <code>true</code> if <code>From</code> is <a href="https://wg21.link/P1144">_TriviallyRelocatable_</a>, to <code>To</code>, aka can be bitwise copied with a facility like <a href="https://en.cppreference.com/w/cpp/string/byte/memcpy"><code>std::memcpy</code></a>, and <code>false</code> otherwise.  */</span><span>template &lt;typename From,</span>
<span>&nbsp;&nbsp;typename To&gt;</span>
<span>constexpr bool <b><a href="/api/groups/group__type__traits.html#variable-is_trivially_relocatable_to_v">type&#95;traits::is&#95;trivially&#95;relocatable&#95;to&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> that is <code>true</code> if the element type of <code>FromIterator</code> is <a href="https://wg21.link/P1144">_TriviallyRelocatable_</a>, to the element type of <code>ToIterator</code>, aka can be bitwise copied with a facility like <a href="https://en.cppreference.com/w/cpp/string/byte/memcpy"><code>std::memcpy</code></a>, and <code>false</code> otherwise.  */</span><span>template &lt;typename FromIterator,</span>
<span>&nbsp;&nbsp;typename ToIterator&gt;</span>
<span>constexpr bool <b><a href="/api/groups/group__type__traits.html#variable-is_indirectly_trivially_relocate_to_v">type&#95;traits::is&#95;indirectly&#95;trivially&#95;relocate&#95;to&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> whose value is <code>(... && Ts::value)</code>.  */</span><span>template &lt;typename... Ts&gt;</span>
<span>constexpr bool <b><a href="/api/groups/group__type__traits.html#variable-conjunction_v">type&#95;traits::conjunction&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> whose value is <code>(... || Ts::value)</code>.  */</span><span>template &lt;typename... Ts&gt;</span>
<span>constexpr bool <b><a href="/api/groups/group__type__traits.html#variable-disjunction_v">type&#95;traits::disjunction&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> whose value is <code>!Ts::value</code>.  */</span><span>template &lt;typename T&gt;</span>
<span>constexpr bool <b><a href="/api/groups/group__type__traits.html#variable-negation_v">type&#95;traits::negation&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> whose value is <code>(... && Bs)</code>.  */</span><span>template &lt;bool... Bs&gt;</span>
<span>constexpr bool <b><a href="/api/groups/group__type__traits.html#variable-conjunction_value_v">type&#95;traits::conjunction&#95;value&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> whose value is <code>(... || Bs)</code>.  */</span><span>template &lt;bool... Bs&gt;</span>
<span>constexpr bool <b><a href="/api/groups/group__type__traits.html#variable-disjunction_value_v">type&#95;traits::disjunction&#95;value&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span class="doxybook-comment">/* <code>constexpr bool</code> whose value is <code>!Ts::value</code>.  */</span><span>template &lt;bool B&gt;</span>
<span>constexpr bool <b><a href="/api/groups/group__type__traits.html#variable-negation_value_v">type&#95;traits::negation&#95;value&#95;v</a></b> = <i>see below</i>;</span>
<br>
<span>#define <b><a href="/api/groups/group__type__traits.html#define-thrust_proclaim_contiguous_iterator">THRUST&#95;PROCLAIM&#95;CONTIGUOUS&#95;ITERATOR</a></b> = <i>see below</i>;</span>
<br>
<span>#define <b><a href="/api/groups/group__type__traits.html#define-thrust_proclaim_trivially_relocatable">THRUST&#95;PROCLAIM&#95;TRIVIALLY&#95;RELOCATABLE</a></b> = <i>see below</i>;</span>
</code>

## Member Classes

<h3 id="struct-proclaim_contiguous_iterator">
<a href="/api/classes/structproclaim__contiguous__iterator.html">Struct <code>proclaim&#95;contiguous&#95;iterator</code>
</a>
</h3>

Customization point that can be customized to indicate that an iterator type <code>Iterator</code> satisfies <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>, aka it points to elements that are contiguous in memory. 

**Inherits From**:
`false_type`

<h3 id="struct-proclaim_trivially_relocatable">
<a href="/api/classes/structproclaim__trivially__relocatable.html">Struct <code>proclaim&#95;trivially&#95;relocatable</code>
</a>
</h3>

<a href="http://eel.is/c++draft/namespace.std#def:customization_point">_customization point_</a> that can be specialized customized to indicate that a type <code>T</code> is <a href="https://wg21.link/P1144">_TriviallyRelocatable_</a>, aka it can be bitwise copied with a facility like <a href="https://en.cppreference.com/w/cpp/string/byte/memcpy"><code>std::memcpy</code></a>. 

**Inherits From**:
`false_type`

<h3 id="struct-conjunction_value">
<a href="/api/classes/structconjunction__value.html">Struct <code>conjunction&#95;value</code>
</a>
</h3>

<a href="https://en.cppreference.com/w/cpp/types/integral_constant"><code>std::integral&#95;constant</code></a> whose value is <code>(... && Bs)</code>. 

<h3 id="struct-disjunction_value">
<a href="/api/classes/structdisjunction__value.html">Struct <code>disjunction&#95;value</code>
</a>
</h3>

<a href="https://en.cppreference.com/w/cpp/types/integral_constant"><code>std::integral&#95;constant</code></a> whose value is <code>(... || Bs)</code>. 

<h3 id="struct-negation_value">
<a href="/api/classes/structnegation__value.html">Struct <code>negation&#95;value</code>
</a>
</h3>

<a href="https://en.cppreference.com/w/cpp/types/integral_constant"><code>std::integral&#95;constant</code></a> whose value is <code>!Bs</code>. 

<h3 id="struct-remove_cvref">
<a href="/api/classes/structremove__cvref.html">Struct <code>remove&#95;cvref</code>
</a>
</h3>

<a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that removes <a href="https://en.cppreference.com/w/cpp/language/cv">const-volatile qualifiers</a> and <a href="https://en.cppreference.com/w/cpp/language/reference">references</a> from <code>T</code>. Equivalent to <code>remove&#95;cv&#95;t&lt;remove&#95;reference&#95;t&lt;T&gt;&gt;</code>. 

<h3 id="struct-voider">
<a href="/api/classes/structvoider.html">Struct <code>voider</code>
</a>
</h3>


## Types

<h3 id="using-integer_sequence">
Type Alias <code>integer&#95;sequence</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;T... Is&gt;</span>
<span>using <b>integer_sequence</b> = std::integer&#95;sequence&lt; T, Is... &gt;;</span></code>
A compile-time sequence of <a href="https://en.cppreference.com/w/cpp/language/constant_expression#Integral_constant_expression">_integral constants_</a> of type <code>T</code> with values <code>Is...</code>. 

**See**:
* <a href="https://en.cppreference.com/w/cpp/language/constant_expression#Integral_constant_expression">_integral constants_</a>
* <a href="/api/groups/group__type__traits.html#using-index_sequence">index_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-make_integer_sequence">make_integer_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-make_reversed_integer_sequence">make_reversed_integer_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-make_index_sequence">make_index_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-make_reversed_index_sequence">make_reversed_index_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-integer_sequence_push_front">integer_sequence_push_front</a>
* <a href="/api/groups/group__type__traits.html#using-integer_sequence_push_back">integer_sequence_push_back</a>
* <a href="https://en.cppreference.com/w/cpp/utility/integer_sequence"><code>std::integer&#95;sequence</code></a>

<h3 id="using-index_sequence">
Type Alias <code>index&#95;sequence</code>
</h3>

<code class="doxybook">
<span>template &lt;std::size_t... Is&gt;</span>
<span>using <b>index_sequence</b> = std::index&#95;sequence&lt; Is... &gt;;</span></code>
A compile-time sequence of type <a href="https://en.cppreference.com/w/cpp/types/size_t">std::size_t</a> with values <code>Is...</code>. 

**See**:
* <a href="/api/groups/group__type__traits.html#using-integer_sequence">integer_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-make_integer_sequence">make_integer_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-make_reversed_integer_sequence">make_reversed_integer_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-make_index_sequence">make_index_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-make_reversed_index_sequence">make_reversed_index_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-integer_sequence_push_front">integer_sequence_push_front</a>
* <a href="/api/groups/group__type__traits.html#using-integer_sequence_push_back">integer_sequence_push_back</a>
* <a href="https://en.cppreference.com/w/cpp/utility/integer_sequence"><code>std::index&#95;sequence</code></a>

<h3 id="using-make_integer_sequence">
Type Alias <code>make&#95;integer&#95;sequence</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;std::size_t N&gt;</span>
<span>using <b>make_integer_sequence</b> = std::make&#95;integer&#95;sequence&lt; T, N &gt;;</span></code>
Create a new <code>integer&#95;sequence</code> with elements <code>0, 1, 2, ..., N - 1</code> of type <code>T</code>. 

**See**:
* <a href="/api/groups/group__type__traits.html#using-integer_sequence">integer_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-index_sequence">index_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-make_reversed_integer_sequence">make_reversed_integer_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-make_index_sequence">make_index_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-make_reversed_index_sequence">make_reversed_index_sequence</a>
* <a href="https://en.cppreference.com/w/cpp/utility/integer_sequence"><code>std::make&#95;integer&#95;sequence</code></a>

<h3 id="using-make_index_sequence">
Type Alias <code>make&#95;index&#95;sequence</code>
</h3>

<code class="doxybook">
<span>template &lt;std::size_t N&gt;</span>
<span>using <b>make_index_sequence</b> = std::make&#95;index&#95;sequence&lt; N &gt;;</span></code>
Create a new <code>integer&#95;sequence</code> with elements <code>0, 1, 2, ..., N - 1</code> of type <a href="https://en.cppreference.com/w/cpp/types/size_t">std::size_t</a>. 

**See**:
* <a href="/api/groups/group__type__traits.html#using-integer_sequence">integer_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-index_sequence">index_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-make_integer_sequence">make_integer_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-make_reversed_integer_sequence">make_reversed_integer_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-make_reversed_index_sequence">make_reversed_index_sequence</a>
* <a href="https://en.cppreference.com/w/cpp/utility/integer_sequence"><code>std::make&#95;index&#95;sequence</code></a>

<h3 id="using-make_reversed_integer_sequence">
Type Alias <code>make&#95;reversed&#95;integer&#95;sequence</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;std::size_t N&gt;</span>
<span>using <b>make_reversed_integer_sequence</b> = typename detail::make&#95;reversed&#95;integer&#95;sequence&#95;impl&lt; T, N &gt;::type;</span></code>
Create a new <code>integer&#95;sequence</code> with elements <code>N - 1, N - 2, N - 3, ..., 0</code>. 

**See**:
* <a href="/api/groups/group__type__traits.html#using-integer_sequence">integer_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-index_sequence">index_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-make_integer_sequence">make_integer_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-make_index_sequence">make_index_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-make_reversed_index_sequence">make_reversed_index_sequence</a>

<h3 id="using-make_reversed_index_sequence">
Type Alias <code>make&#95;reversed&#95;index&#95;sequence</code>
</h3>

<code class="doxybook">
<span>template &lt;std::size_t N&gt;</span>
<span>using <b>make_reversed_index_sequence</b> = &lt;a href="/api/groups/group&#95;&#95;type&#95;&#95;traits.html#using-make&#95;reversed&#95;integer&#95;sequence"&gt;make&#95;reversed&#95;integer&#95;sequence&lt;/a&gt;&lt; std::size&#95;t, N &gt;;</span></code>
Create a new <code>index&#95;sequence</code> with elements <code>N - 1, N - 2, N - 3, ..., 0</code>. 

**See**:
* <a href="/api/groups/group__type__traits.html#using-integer_sequence">integer_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-index_sequence">index_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-make_integer_sequence">make_integer_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-make_reversed_integer_sequence">make_reversed_integer_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-make_reversed_index_sequence">make_reversed_index_sequence</a>

<h3 id="using-integer_sequence_push_front">
Type Alias <code>integer&#95;sequence&#95;push&#95;front</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;T Value,</span>
<span>&nbsp;&nbsp;typename Sequence&gt;</span>
<span>using <b>integer_sequence_push_front</b> = typename detail::integer&#95;sequence&#95;push&#95;front&#95;impl&lt; T, Value, Sequence &gt;::type;</span></code>
Add a new element to the front of an <code>integer&#95;sequence</code>. 

**See**:
* <a href="/api/groups/group__type__traits.html#using-integer_sequence">integer_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-index_sequence">index_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-make_integer_sequence">make_integer_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-make_index_sequence">make_index_sequence</a>

<h3 id="using-integer_sequence_push_back">
Type Alias <code>integer&#95;sequence&#95;push&#95;back</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T,</span>
<span>&nbsp;&nbsp;T Value,</span>
<span>&nbsp;&nbsp;typename Sequence&gt;</span>
<span>using <b>integer_sequence_push_back</b> = typename detail::integer&#95;sequence&#95;push&#95;back&#95;impl&lt; T, Value, Sequence &gt;::type;</span></code>
Add a new element to the back of an <code>integer&#95;sequence</code>. 

**See**:
* <a href="/api/groups/group__type__traits.html#using-integer_sequence">integer_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-index_sequence">index_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-make_integer_sequence">make_integer_sequence</a>
* <a href="/api/groups/group__type__traits.html#using-make_index_sequence">make_index_sequence</a>

<h3 id="using-is_contiguous_iterator">
Type Alias <code>is&#95;contiguous&#95;iterator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Iterator&gt;</span>
<span>using <b>is_contiguous_iterator</b> = detail::is&#95;contiguous&#95;iterator&#95;impl&lt; Iterator &gt;;</span></code>
<a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>Iterator</code> satisfies <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>, aka it points to elements that are contiguous in memory, and <code>false&#95;type</code> otherwise. 

**See**:
* <a href="/api/groups/group__type__traits.html#variable-is_contiguous_iterator_v">is_contiguous_iterator_v</a>
* <a href="/api/classes/structproclaim__contiguous__iterator.html">proclaim_contiguous_iterator</a>
* <a href="/api/groups/group__type__traits.html#define-thrust_proclaim_contiguous_iterator">THRUST_PROCLAIM_CONTIGUOUS_ITERATOR</a>

<h3 id="using-is_execution_policy">
Type Alias <code>is&#95;execution&#95;policy</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>is_execution_policy</b> = detail::is&#95;base&#95;of&lt; detail::execution&#95;policy&#95;marker, T &gt;;</span></code>
<a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>T</code> is an _ExecutionPolicy_ and <code>false&#95;type</code> otherwise. 

<h3 id="using-is_operator_less_function_object">
Type Alias <code>is&#95;operator&#95;less&#95;function&#95;object</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>is_operator_less_function_object</b> = detail::is&#95;operator&#95;less&#95;function&#95;object&#95;impl&lt; T &gt;;</span></code>
<a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a> equivalent to <code>operator&lt;</code>, and <code>false&#95;type</code> otherwise. 

**See**:
* <a href="/api/groups/group__type__traits.html#variable-is_operator_less_function_object_v">is_operator_less_function_object_v</a>
* <a href="/api/groups/group__type__traits.html#using-is_operator_greater_function_object">is_operator_greater_function_object</a>
* <a href="/api/groups/group__type__traits.html#using-is_operator_less_or_greater_function_object">is_operator_less_or_greater_function_object</a>
* <a href="/api/groups/group__type__traits.html#using-is_operator_plus_function_object">is_operator_plus_function_object</a>

<h3 id="using-is_operator_greater_function_object">
Type Alias <code>is&#95;operator&#95;greater&#95;function&#95;object</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>is_operator_greater_function_object</b> = detail::is&#95;operator&#95;greater&#95;function&#95;object&#95;impl&lt; T &gt;;</span></code>
<a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a> equivalent to <code>operator&gt;</code>, and <code>false&#95;type</code> otherwise. 

**See**:
* <a href="/api/groups/group__type__traits.html#variable-is_operator_greater_function_object_v">is_operator_greater_function_object_v</a>
* <a href="/api/groups/group__type__traits.html#using-is_operator_less_function_object">is_operator_less_function_object</a>
* <a href="/api/groups/group__type__traits.html#using-is_operator_less_or_greater_function_object">is_operator_less_or_greater_function_object</a>
* <a href="/api/groups/group__type__traits.html#using-is_operator_plus_function_object">is_operator_plus_function_object</a>

<h3 id="using-is_operator_less_or_greater_function_object">
Type Alias <code>is&#95;operator&#95;less&#95;or&#95;greater&#95;function&#95;object</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>is_operator_less_or_greater_function_object</b> = integral&#95;constant&lt; bool, detail::is&#95;operator&#95;less&#95;function&#95;object&#95;impl&lt; T &gt;::value||detail::is&#95;operator&#95;greater&#95;function&#95;object&#95;impl&lt; T &gt;::value &gt;;</span></code>
<a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a> equivalent to <code>operator&lt;</code> or <code>operator&gt;</code>, and <code>false&#95;type</code> otherwise. 

**See**:
* <a href="/api/groups/group__type__traits.html#variable-is_operator_less_or_greater_function_object_v">is_operator_less_or_greater_function_object_v</a>
* <a href="/api/groups/group__type__traits.html#using-is_operator_less_function_object">is_operator_less_function_object</a>
* <a href="/api/groups/group__type__traits.html#using-is_operator_greater_function_object">is_operator_greater_function_object</a>
* <a href="/api/groups/group__type__traits.html#using-is_operator_plus_function_object">is_operator_plus_function_object</a>

<h3 id="using-is_operator_plus_function_object">
Type Alias <code>is&#95;operator&#95;plus&#95;function&#95;object</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>is_operator_plus_function_object</b> = detail::is&#95;operator&#95;plus&#95;function&#95;object&#95;impl&lt; T &gt;;</span></code>
<a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/FunctionObject">FunctionObject</a> equivalent to <code>operator+</code>, and <code>false&#95;type</code> otherwise. 

**See**:
* <a href="/api/groups/group__type__traits.html#variable-is_operator_plus_function_object_v">is_operator_plus_function_object_v</a>
* <a href="/api/groups/group__type__traits.html#using-is_operator_less_function_object">is_operator_less_function_object</a>
* <a href="/api/groups/group__type__traits.html#using-is_operator_greater_function_object">is_operator_greater_function_object</a>
* <a href="/api/groups/group__type__traits.html#using-is_operator_less_or_greater_function_object">is_operator_less_or_greater_function_object</a>

<h3 id="using-is_trivially_relocatable">
Type Alias <code>is&#95;trivially&#95;relocatable</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>is_trivially_relocatable</b> = detail::is&#95;trivially&#95;relocatable&#95;impl&lt; T &gt;;</span></code>
<a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait">_UnaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>T</code> is <a href="https://wg21.link/P1144">_TriviallyRelocatable_</a>, aka can be bitwise copied with a facility like <a href="https://en.cppreference.com/w/cpp/string/byte/memcpy"><code>std::memcpy</code></a>, and <code>false&#95;type</code> otherwise. 

**See**:
* <a href="/api/groups/group__type__traits.html#variable-is_trivially_relocatable_v">is_trivially_relocatable_v</a>
* <a href="/api/groups/group__type__traits.html#using-is_trivially_relocatable_to">is_trivially_relocatable_to</a>
* <a href="/api/groups/group__type__traits.html#using-is_indirectly_trivially_relocatable_to">is_indirectly_trivially_relocatable_to</a>
* <a href="/api/classes/structproclaim__trivially__relocatable.html">proclaim_trivially_relocatable</a>
* <a href="/api/groups/group__type__traits.html#define-thrust_proclaim_trivially_relocatable">THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE</a>

<h3 id="using-is_trivially_relocatable_to">
Type Alias <code>is&#95;trivially&#95;relocatable&#95;to</code>
</h3>

<code class="doxybook">
<span>template &lt;typename From,</span>
<span>&nbsp;&nbsp;typename To&gt;</span>
<span>using <b>is_trivially_relocatable_to</b> = integral&#95;constant&lt; bool, detail::is&#95;same&lt; From, To &gt;::value &&&lt;a href="/api/groups/group&#95;&#95;type&#95;&#95;traits.html#using-is&#95;trivially&#95;relocatable"&gt;is&#95;trivially&#95;relocatable&lt;/a&gt;&lt; To &gt;::value &gt;;</span></code>
<a href="https://en.cppreference.com/w/cpp/named_req/BinaryTypeTrait">_BinaryTypeTrait_</a> that returns <code>true&#95;type</code> if <code>From</code> is <a href="https://wg21.link/P1144">_TriviallyRelocatable_</a>, to <code>To</code>, aka can be bitwise copied with a facility like <a href="https://en.cppreference.com/w/cpp/string/byte/memcpy"><code>std::memcpy</code></a>, and <code>false&#95;type</code> otherwise. 

**See**:
* <a href="/api/groups/group__type__traits.html#variable-is_trivially_relocatable_to_v">is_trivially_relocatable_to_v</a>
* <a href="/api/groups/group__type__traits.html#using-is_trivially_relocatable">is_trivially_relocatable</a>
* <a href="/api/groups/group__type__traits.html#using-is_indirectly_trivially_relocatable_to">is_indirectly_trivially_relocatable_to</a>
* <a href="/api/classes/structproclaim__trivially__relocatable.html">proclaim_trivially_relocatable</a>
* <a href="/api/groups/group__type__traits.html#define-thrust_proclaim_trivially_relocatable">THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE</a>

<h3 id="using-is_indirectly_trivially_relocatable_to">
Type Alias <code>is&#95;indirectly&#95;trivially&#95;relocatable&#95;to</code>
</h3>

<code class="doxybook">
<span>template &lt;typename FromIterator,</span>
<span>&nbsp;&nbsp;typename ToIterator&gt;</span>
<span>using <b>is_indirectly_trivially_relocatable_to</b> = integral&#95;constant&lt; bool, &lt;a href="/api/groups/group&#95;&#95;type&#95;&#95;traits.html#using-is&#95;contiguous&#95;iterator"&gt;is&#95;contiguous&#95;iterator&lt;/a&gt;&lt; FromIterator &gt;::value &&&lt;a href="/api/groups/group&#95;&#95;type&#95;&#95;traits.html#using-is&#95;contiguous&#95;iterator"&gt;is&#95;contiguous&#95;iterator&lt;/a&gt;&lt; ToIterator &gt;::value &&&lt;a href="/api/groups/group&#95;&#95;type&#95;&#95;traits.html#using-is&#95;trivially&#95;relocatable&#95;to"&gt;is&#95;trivially&#95;relocatable&#95;to&lt;/a&gt;&lt; typename thrust::iterator&#95;traits&lt; FromIterator &gt;::value&#95;type, typename thrust::iterator&#95;traits&lt; ToIterator &gt;::value&#95;type &gt;::value &gt;;</span></code>
<a href="https://en.cppreference.com/w/cpp/named_req/BinaryTypeTrait">_BinaryTypeTrait_</a> that returns <code>true&#95;type</code> if the element type of <code>FromIterator</code> is <a href="https://wg21.link/P1144">_TriviallyRelocatable_</a>, to the element type of <code>ToIterator</code>, aka can be bitwise copied with a facility like <a href="https://en.cppreference.com/w/cpp/string/byte/memcpy"><code>std::memcpy</code></a>, and <code>false&#95;type</code> otherwise. 

**See**:
* is_indirectly_trivially_relocatable_to_v 
* <a href="/api/groups/group__type__traits.html#using-is_trivially_relocatable">is_trivially_relocatable</a>
* <a href="/api/groups/group__type__traits.html#using-is_trivially_relocatable_to">is_trivially_relocatable_to</a>
* <a href="/api/classes/structproclaim__trivially__relocatable.html">proclaim_trivially_relocatable</a>
* <a href="/api/groups/group__type__traits.html#define-thrust_proclaim_trivially_relocatable">THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE</a>

<h3 id="using-conjunction">
Type Alias <code>conjunction</code>
</h3>

<code class="doxybook">
<span>template &lt;typename... Ts&gt;</span>
<span>using <b>conjunction</b> = std::conjunction&lt; Ts... &gt;;</span></code>
<a href="https://en.cppreference.com/w/cpp/types/integral_constant"><code>std::integral&#95;constant</code></a> whose value is <code>(... && Ts::value)</code>. 

**See**:
* <a href="/api/groups/group__type__traits.html#variable-conjunction_v">conjunction_v</a>
* <a href="/api/classes/structconjunction__value.html">conjunction_value</a>
* <a href="https://en.cppreference.com/w/cpp/types/conjunction"><code>std::conjunction</code></a>

<h3 id="using-disjunction">
Type Alias <code>disjunction</code>
</h3>

<code class="doxybook">
<span>template &lt;typename... Ts&gt;</span>
<span>using <b>disjunction</b> = std::disjunction&lt; Ts... &gt;;</span></code>
<a href="https://en.cppreference.com/w/cpp/types/integral_constant"><code>std::integral&#95;constant</code></a> whose value is <code>(... || Ts::value)</code>. 

**See**:
* <a href="/api/groups/group__type__traits.html#variable-disjunction_v">disjunction_v</a>
* <a href="/api/classes/structdisjunction__value.html">disjunction_value</a>
* <a href="https://en.cppreference.com/w/cpp/types/disjunction"><code>std::disjunction</code></a>

<h3 id="using-negation">
Type Alias <code>negation</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>negation</b> = std::negation&lt; T &gt;;</span></code>
<a href="https://en.cppreference.com/w/cpp/types/integral_constant"><code>std::integral&#95;constant</code></a> whose value is <code>!Ts::value</code>. 

**See**:
* <a href="/api/groups/group__type__traits.html#variable-negation_v">negation_v</a>
* <a href="/api/classes/structnegation__value.html">negation_value</a>
* <a href="https://en.cppreference.com/w/cpp/types/negation"><code>std::negation</code></a>

<h3 id="using-remove_cvref_t">
Type Alias <code>remove&#95;cvref&#95;t</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>using <b>remove_cvref_t</b> = typename &lt;a href="/api/classes/structremove&#95;&#95;cvref.html"&gt;remove&#95;cvref&lt;/a&gt;&lt; T &gt;::type;</span></code>
Type alias that removes <a href="https://en.cppreference.com/w/cpp/language/cv">const-volatile qualifiers</a> and <a href="https://en.cppreference.com/w/cpp/language/reference">references</a> from <code>T</code>. Equivalent to <code>remove&#95;cv&#95;t&lt;remove&#95;reference&#95;t&lt;T&gt;&gt;</code>. 

**See**:
* <a href="https://en.cppreference.com/w/cpp/types/remove_cvref">std::remove_cvref</a>
* <a href="https://en.cppreference.com/w/cpp/types/remove_cv">std::remove_cv</a>
* <a href="https://en.cppreference.com/w/cpp/types/remove_cv">std::remove_const</a>
* <a href="https://en.cppreference.com/w/cpp/types/remove_cv">std::remove_volatile</a>
* <a href="https://en.cppreference.com/w/cpp/types/remove_cv">std::remove_reference</a>


## Variables

<h3 id="variable-is_contiguous_iterator_v">
Variable <code>type&#95;traits::is&#95;contiguous&#95;iterator&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Iterator&gt;</span>
<span>constexpr bool <b>is_contiguous_iterator_v</b> = &lt;a href="/api/groups/group&#95;&#95;type&#95;&#95;traits.html#using-is&#95;contiguous&#95;iterator"&gt;is&#95;contiguous&#95;iterator&lt;/a&gt;&lt;Iterator&gt;::value;</span></code>
<code>constexpr bool</code> that is <code>true</code> if <code>Iterator</code> satisfies <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>, aka it points to elements that are contiguous in memory, and <code>false</code> otherwise. 

**See**:
* <a href="/api/groups/group__type__traits.html#using-is_contiguous_iterator">is_contiguous_iterator</a>
* <a href="/api/classes/structproclaim__contiguous__iterator.html">proclaim_contiguous_iterator</a>
* <a href="/api/groups/group__type__traits.html#define-thrust_proclaim_contiguous_iterator">THRUST_PROCLAIM_CONTIGUOUS_ITERATOR</a>

<h3 id="variable-is_execution_policy_v">
Variable <code>type&#95;traits::is&#95;execution&#95;policy&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>constexpr bool <b>is_execution_policy_v</b> = &lt;a href="/api/groups/group&#95;&#95;type&#95;&#95;traits.html#using-is&#95;execution&#95;policy"&gt;is&#95;execution&#95;policy&lt;/a&gt;&lt;T&gt;::value;</span></code>
<code>constexpr bool</code> that is <code>true</code> if <code>T</code> is an _ExecutionPolicy_ and <code>false</code> otherwise. 

<h3 id="variable-is_operator_less_function_object_v">
Variable <code>type&#95;traits::is&#95;operator&#95;less&#95;function&#95;object&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>constexpr bool <b>is_operator_less_function_object_v</b> = &lt;a href="/api/groups/group&#95;&#95;type&#95;&#95;traits.html#using-is&#95;operator&#95;less&#95;function&#95;object"&gt;is&#95;operator&#95;less&#95;function&#95;object&lt;/a&gt;&lt;T&gt;::value;</span></code>
<code>constexpr bool</code> that is <code>true</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a> equivalent to <code>operator&lt;</code>, and <code>false</code> otherwise. 

**See**:
* <a href="/api/groups/group__type__traits.html#using-is_operator_less_function_object">is_operator_less_function_object</a>
* <a href="/api/groups/group__type__traits.html#using-is_operator_greater_function_object">is_operator_greater_function_object</a>
* <a href="/api/groups/group__type__traits.html#using-is_operator_less_or_greater_function_object">is_operator_less_or_greater_function_object</a>
* <a href="/api/groups/group__type__traits.html#using-is_operator_plus_function_object">is_operator_plus_function_object</a>

<h3 id="variable-is_operator_greater_function_object_v">
Variable <code>type&#95;traits::is&#95;operator&#95;greater&#95;function&#95;object&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>constexpr bool <b>is_operator_greater_function_object_v</b> = &lt;a href="/api/groups/group&#95;&#95;type&#95;&#95;traits.html#using-is&#95;operator&#95;greater&#95;function&#95;object"&gt;is&#95;operator&#95;greater&#95;function&#95;object&lt;/a&gt;&lt;T&gt;::value;</span></code>
<code>constexpr bool</code> that is <code>true</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a> equivalent to <code>operator&gt;</code>, and <code>false</code> otherwise. 

**See**:
* <a href="/api/groups/group__type__traits.html#using-is_operator_greater_function_object">is_operator_greater_function_object</a>
* <a href="/api/groups/group__type__traits.html#using-is_operator_less_function_object">is_operator_less_function_object</a>
* <a href="/api/groups/group__type__traits.html#using-is_operator_less_or_greater_function_object">is_operator_less_or_greater_function_object</a>
* <a href="/api/groups/group__type__traits.html#using-is_operator_plus_function_object">is_operator_plus_function_object</a>

<h3 id="variable-is_operator_less_or_greater_function_object_v">
Variable <code>type&#95;traits::is&#95;operator&#95;less&#95;or&#95;greater&#95;function&#95;object&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>constexpr bool <b>is_operator_less_or_greater_function_object_v</b> = &lt;a href="/api/groups/group&#95;&#95;type&#95;&#95;traits.html#using-is&#95;operator&#95;less&#95;or&#95;greater&#95;function&#95;object"&gt;is&#95;operator&#95;less&#95;or&#95;greater&#95;function&#95;object&lt;/a&gt;&lt;T&gt;::value;</span></code>
<code>constexpr bool</code> that is <code>true</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">BinaryPredicate</a> equivalent to <code>operator&lt;</code> or <code>operator&gt;</code>, and <code>false</code> otherwise. 

**See**:
* <a href="/api/groups/group__type__traits.html#using-is_operator_less_or_greater_function_object">is_operator_less_or_greater_function_object</a>
* <a href="/api/groups/group__type__traits.html#using-is_operator_less_function_object">is_operator_less_function_object</a>
* <a href="/api/groups/group__type__traits.html#using-is_operator_greater_function_object">is_operator_greater_function_object</a>
* <a href="/api/groups/group__type__traits.html#using-is_operator_plus_function_object">is_operator_plus_function_object</a>

<h3 id="variable-is_operator_plus_function_object_v">
Variable <code>type&#95;traits::is&#95;operator&#95;plus&#95;function&#95;object&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>constexpr bool <b>is_operator_plus_function_object_v</b> = &lt;a href="/api/groups/group&#95;&#95;type&#95;&#95;traits.html#using-is&#95;operator&#95;plus&#95;function&#95;object"&gt;is&#95;operator&#95;plus&#95;function&#95;object&lt;/a&gt;&lt;T&gt;::value;</span></code>
<code>constexpr bool</code> that is <code>true</code> if <code>T</code> is a <a href="https://en.cppreference.com/w/cpp/named_req/FunctionObject">FunctionObject</a> equivalent to <code>operator&lt;</code>, and <code>false</code> otherwise. 

**See**:
* <a href="/api/groups/group__type__traits.html#using-is_operator_plus_function_object">is_operator_plus_function_object</a>
* <a href="/api/groups/group__type__traits.html#using-is_operator_less_function_object">is_operator_less_function_object</a>
* <a href="/api/groups/group__type__traits.html#using-is_operator_greater_function_object">is_operator_greater_function_object</a>
* <a href="/api/groups/group__type__traits.html#using-is_operator_less_or_greater_function_object">is_operator_less_or_greater_function_object</a>

<h3 id="variable-is_trivially_relocatable_v">
Variable <code>type&#95;traits::is&#95;trivially&#95;relocatable&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>constexpr bool <b>is_trivially_relocatable_v</b> = &lt;a href="/api/groups/group&#95;&#95;type&#95;&#95;traits.html#using-is&#95;trivially&#95;relocatable"&gt;is&#95;trivially&#95;relocatable&lt;/a&gt;&lt;T&gt;::value;</span></code>
<code>constexpr bool</code> that is <code>true</code> if <code>T</code> is <a href="https://wg21.link/P1144">_TriviallyRelocatable_</a>, aka can be bitwise copied with a facility like <a href="https://en.cppreference.com/w/cpp/string/byte/memcpy"><code>std::memcpy</code></a>, and <code>false</code> otherwise. 

**See**:
* <a href="/api/groups/group__type__traits.html#using-is_trivially_relocatable">is_trivially_relocatable</a>
* <a href="/api/groups/group__type__traits.html#using-is_trivially_relocatable_to">is_trivially_relocatable_to</a>
* <a href="/api/groups/group__type__traits.html#using-is_indirectly_trivially_relocatable_to">is_indirectly_trivially_relocatable_to</a>
* <a href="/api/classes/structproclaim__trivially__relocatable.html">proclaim_trivially_relocatable</a>
* <a href="/api/groups/group__type__traits.html#define-thrust_proclaim_trivially_relocatable">THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE</a>

<h3 id="variable-is_trivially_relocatable_to_v">
Variable <code>type&#95;traits::is&#95;trivially&#95;relocatable&#95;to&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;typename From,</span>
<span>&nbsp;&nbsp;typename To&gt;</span>
<span>constexpr bool <b>is_trivially_relocatable_to_v</b> = &lt;a href="/api/groups/group&#95;&#95;type&#95;&#95;traits.html#using-is&#95;trivially&#95;relocatable&#95;to"&gt;is&#95;trivially&#95;relocatable&#95;to&lt;/a&gt;&lt;From, To&gt;::value;</span></code>
<code>constexpr bool</code> that is <code>true</code> if <code>From</code> is <a href="https://wg21.link/P1144">_TriviallyRelocatable_</a>, to <code>To</code>, aka can be bitwise copied with a facility like <a href="https://en.cppreference.com/w/cpp/string/byte/memcpy"><code>std::memcpy</code></a>, and <code>false</code> otherwise. 

**See**:
* <a href="/api/groups/group__type__traits.html#using-is_trivially_relocatable_to">is_trivially_relocatable_to</a>
* <a href="/api/groups/group__type__traits.html#using-is_trivially_relocatable">is_trivially_relocatable</a>
* <a href="/api/groups/group__type__traits.html#using-is_indirectly_trivially_relocatable_to">is_indirectly_trivially_relocatable_to</a>
* <a href="/api/classes/structproclaim__trivially__relocatable.html">proclaim_trivially_relocatable</a>
* <a href="/api/groups/group__type__traits.html#define-thrust_proclaim_trivially_relocatable">THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE</a>

<h3 id="variable-is_indirectly_trivially_relocate_to_v">
Variable <code>type&#95;traits::is&#95;indirectly&#95;trivially&#95;relocate&#95;to&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;typename FromIterator,</span>
<span>&nbsp;&nbsp;typename ToIterator&gt;</span>
<span>constexpr bool <b>is_indirectly_trivially_relocate_to_v</b> = &lt;a href="/api/groups/group&#95;&#95;type&#95;&#95;traits.html#using-is&#95;indirectly&#95;trivially&#95;relocatable&#95;to"&gt;is&#95;indirectly&#95;trivially&#95;relocatable&#95;to&lt;/a&gt;&lt;FromIterator, ToIterator&gt;::value;</span></code>
<code>constexpr bool</code> that is <code>true</code> if the element type of <code>FromIterator</code> is <a href="https://wg21.link/P1144">_TriviallyRelocatable_</a>, to the element type of <code>ToIterator</code>, aka can be bitwise copied with a facility like <a href="https://en.cppreference.com/w/cpp/string/byte/memcpy"><code>std::memcpy</code></a>, and <code>false</code> otherwise. 

**See**:
* <a href="/api/groups/group__type__traits.html#using-is_indirectly_trivially_relocatable_to">is_indirectly_trivially_relocatable_to</a>
* <a href="/api/groups/group__type__traits.html#using-is_trivially_relocatable">is_trivially_relocatable</a>
* <a href="/api/groups/group__type__traits.html#using-is_trivially_relocatable_to">is_trivially_relocatable_to</a>
* <a href="/api/classes/structproclaim__trivially__relocatable.html">proclaim_trivially_relocatable</a>
* <a href="/api/groups/group__type__traits.html#define-thrust_proclaim_trivially_relocatable">THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE</a>

<h3 id="variable-conjunction_v">
Variable <code>type&#95;traits::conjunction&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;typename... Ts&gt;</span>
<span>constexpr bool <b>conjunction_v</b> = &lt;a href="/api/groups/group&#95;&#95;type&#95;&#95;traits.html#using-conjunction"&gt;conjunction&lt;/a&gt;&lt;Ts...&gt;::value;</span></code>
<code>constexpr bool</code> whose value is <code>(... && Ts::value)</code>. 

**See**:
* <a href="/api/groups/group__type__traits.html#using-conjunction">conjunction</a>
* <a href="/api/classes/structconjunction__value.html">conjunction_value</a>
* <a href="https://en.cppreference.com/w/cpp/types/conjunction"><code>std::conjunction</code></a>

<h3 id="variable-disjunction_v">
Variable <code>type&#95;traits::disjunction&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;typename... Ts&gt;</span>
<span>constexpr bool <b>disjunction_v</b> = &lt;a href="/api/groups/group&#95;&#95;type&#95;&#95;traits.html#using-disjunction"&gt;disjunction&lt;/a&gt;&lt;Ts...&gt;::value;</span></code>
<code>constexpr bool</code> whose value is <code>(... || Ts::value)</code>. 

**See**:
* <a href="/api/groups/group__type__traits.html#using-disjunction">disjunction</a>
* <a href="/api/classes/structdisjunction__value.html">disjunction_value</a>
* <a href="https://en.cppreference.com/w/cpp/types/disjunction"><code>std::disjunction</code></a>

<h3 id="variable-negation_v">
Variable <code>type&#95;traits::negation&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;typename T&gt;</span>
<span>constexpr bool <b>negation_v</b> = &lt;a href="/api/groups/group&#95;&#95;type&#95;&#95;traits.html#using-negation"&gt;negation&lt;/a&gt;&lt;T&gt;::value;</span></code>
<code>constexpr bool</code> whose value is <code>!Ts::value</code>. 

**See**:
* <a href="/api/groups/group__type__traits.html#using-negation">negation</a>
* <a href="/api/classes/structnegation__value.html">negation_value</a>
* <a href="https://en.cppreference.com/w/cpp/types/negation"><code>std::negation</code></a>

<h3 id="variable-conjunction_value_v">
Variable <code>type&#95;traits::conjunction&#95;value&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;bool... Bs&gt;</span>
<span>constexpr bool <b>conjunction_value_v</b> = &lt;a href="/api/classes/structconjunction&#95;&#95;value.html"&gt;conjunction&#95;value&lt;/a&gt;&lt;Bs...&gt;::value;</span></code>
<code>constexpr bool</code> whose value is <code>(... && Bs)</code>. 

**See**:
* <a href="/api/classes/structconjunction__value.html">conjunction_value</a>
* <a href="/api/groups/group__type__traits.html#using-conjunction">conjunction</a>
* <a href="https://en.cppreference.com/w/cpp/types/conjunction"><code>std::conjunction</code></a>

<h3 id="variable-disjunction_value_v">
Variable <code>type&#95;traits::disjunction&#95;value&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;bool... Bs&gt;</span>
<span>constexpr bool <b>disjunction_value_v</b> = &lt;a href="/api/classes/structdisjunction&#95;&#95;value.html"&gt;disjunction&#95;value&lt;/a&gt;&lt;Bs...&gt;::value;</span></code>
<code>constexpr bool</code> whose value is <code>(... || Bs)</code>. 

**See**:
* <a href="/api/classes/structdisjunction__value.html">disjunction_value</a>
* <a href="/api/groups/group__type__traits.html#using-disjunction">disjunction</a>
* <a href="https://en.cppreference.com/w/cpp/types/disjunction"><code>std::disjunction</code></a>

<h3 id="variable-negation_value_v">
Variable <code>type&#95;traits::negation&#95;value&#95;v</code>
</h3>

<code class="doxybook">
<span>template &lt;bool B&gt;</span>
<span>constexpr bool <b>negation_value_v</b> = &lt;a href="/api/classes/structnegation&#95;&#95;value.html"&gt;negation&#95;value&lt;/a&gt;&lt;B&gt;::value;</span></code>
<code>constexpr bool</code> whose value is <code>!Ts::value</code>. 

**See**:
* <a href="/api/classes/structnegation__value.html">negation_value</a>
* <a href="/api/groups/group__type__traits.html#using-negation">negation</a>
* <a href="https://en.cppreference.com/w/cpp/types/negation"><code>std::negation</code></a>


## Macros

<h3 id="define-THRUST_PROCLAIM_CONTIGUOUS_ITERATOR">
Define <code>THRUST&#95;PROCLAIM&#95;CONTIGUOUS&#95;ITERATOR</code>
</h3>

<code class="doxybook">
  <span>#define <b>THRUST_PROCLAIM_CONTIGUOUS_ITERATOR</b>   THRUST&#95;NAMESPACE&#95;BEGIN                                                      \
  template &lt;&gt;                                                                 \
  struct &lt;a href="/api/classes/structproclaim&#95;&#95;contiguous&#95;&#95;iterator.html"&gt;proclaim&#95;contiguous&#95;iterator&lt;/a&gt;&lt;Iterator&gt;                               \
      : THRUST&#95;NS&#95;QUALIFIER::true&#95;type {};                                    \
  THRUST&#95;NAMESPACE&#95;END                                                        \;</span></code>
Declares that the iterator <code>Iterator</code> is <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a> by specializing <code><a href="/api/classes/structproclaim__contiguous__iterator.html">proclaim&#95;contiguous&#95;iterator</a></code>. 

**See**:
* <a href="/api/groups/group__type__traits.html#using-is_contiguous_iterator">is_contiguous_iterator</a>
* <a href="/api/classes/structproclaim__contiguous__iterator.html">proclaim_contiguous_iterator</a>

<h3 id="define-THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE">
Define <code>THRUST&#95;PROCLAIM&#95;TRIVIALLY&#95;RELOCATABLE</code>
</h3>

<code class="doxybook">
  <span>#define <b>THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE</b>   THRUST&#95;NAMESPACE&#95;BEGIN                                                      \
  template &lt;&gt;                                                                 \
  struct &lt;a href="/api/classes/structproclaim&#95;&#95;trivially&#95;&#95;relocatable.html"&gt;proclaim&#95;trivially&#95;relocatable&lt;/a&gt;&lt;T&gt; : THRUST&#95;NS&#95;QUALIFIER::true&#95;type   \
  {};                                                                         \
  THRUST&#95;NAMESPACE&#95;END                                                        \;</span></code>
Declares that the type <code>T</code> is <a href="https://wg21.link/P1144">_TriviallyRelocatable_</a>, aka it can be bitwise copied with a facility like <a href="https://en.cppreference.com/w/cpp/string/byte/memcpy"><code>std::memcpy</code></a>, by specializing <code><a href="/api/classes/structproclaim__trivially__relocatable.html">proclaim&#95;trivially&#95;relocatable</a></code>. 

**See**:
* <a href="/api/groups/group__type__traits.html#using-is_indirectly_trivially_relocatable_to">is_indirectly_trivially_relocatable_to</a>
* <a href="/api/groups/group__type__traits.html#using-is_trivially_relocatable">is_trivially_relocatable</a>
* <a href="/api/groups/group__type__traits.html#using-is_trivially_relocatable_to">is_trivially_relocatable_to</a>
* <a href="/api/classes/structproclaim__trivially__relocatable.html">proclaim_trivially_relocatable</a>


