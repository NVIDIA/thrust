---
title: Iterator Tag Classes
parent: Iterator Tags
grand_parent: Iterators
nav_exclude: false
has_children: true
has_toc: false
---

# Iterator Tag Classes

<code class="doxybook">
<span>struct <b><a href="/thrust/api/classes/structinput__device__iterator__tag.html">input&#95;device&#95;iterator&#95;tag</a></b>;</span>
<br>
<span>struct <b><a href="/thrust/api/classes/structoutput__device__iterator__tag.html">output&#95;device&#95;iterator&#95;tag</a></b>;</span>
<br>
<span>struct <b><a href="/thrust/api/classes/structforward__device__iterator__tag.html">forward&#95;device&#95;iterator&#95;tag</a></b>;</span>
<br>
<span>struct <b><a href="/thrust/api/classes/structbidirectional__device__iterator__tag.html">bidirectional&#95;device&#95;iterator&#95;tag</a></b>;</span>
<br>
<span>struct <b><a href="/thrust/api/classes/structrandom__access__device__iterator__tag.html">random&#95;access&#95;device&#95;iterator&#95;tag</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="/thrust/api/groups/group__iterator__tag__classes.html#typedef-input_host_iterator_tag">input&#95;host&#95;iterator&#95;tag</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="/thrust/api/groups/group__iterator__tag__classes.html#typedef-output_host_iterator_tag">output&#95;host&#95;iterator&#95;tag</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="/thrust/api/groups/group__iterator__tag__classes.html#typedef-forward_host_iterator_tag">forward&#95;host&#95;iterator&#95;tag</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="/thrust/api/groups/group__iterator__tag__classes.html#typedef-bidirectional_host_iterator_tag">bidirectional&#95;host&#95;iterator&#95;tag</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="/thrust/api/groups/group__iterator__tag__classes.html#typedef-random_access_host_iterator_tag">random&#95;access&#95;host&#95;iterator&#95;tag</a></b>;</span>
</code>

## Member Classes

<h3 id="struct-input_device_iterator_tag">
<a href="/thrust/api/classes/structinput__device__iterator__tag.html">Struct <code>input&#95;device&#95;iterator&#95;tag</code>
</a>
</h3>

**Inherits From**:
`thrust::detail::iterator_category_with_system_and_traversal< std::input_iterator_tag, thrust::device_system_tag, thrust::single_pass_traversal_tag >`

<h3 id="struct-output_device_iterator_tag">
<a href="/thrust/api/classes/structoutput__device__iterator__tag.html">Struct <code>output&#95;device&#95;iterator&#95;tag</code>
</a>
</h3>

**Inherits From**:
`thrust::detail::iterator_category_with_system_and_traversal< std::output_iterator_tag, thrust::device_system_tag, thrust::single_pass_traversal_tag >`

<h3 id="struct-forward_device_iterator_tag">
<a href="/thrust/api/classes/structforward__device__iterator__tag.html">Struct <code>forward&#95;device&#95;iterator&#95;tag</code>
</a>
</h3>

**Inherits From**:
`thrust::detail::iterator_category_with_system_and_traversal< std::forward_iterator_tag, thrust::device_system_tag, thrust::forward_traversal_tag >`

<h3 id="struct-bidirectional_device_iterator_tag">
<a href="/thrust/api/classes/structbidirectional__device__iterator__tag.html">Struct <code>bidirectional&#95;device&#95;iterator&#95;tag</code>
</a>
</h3>

**Inherits From**:
`thrust::detail::iterator_category_with_system_and_traversal< std::bidirectional_iterator_tag, thrust::device_system_tag, thrust::bidirectional_traversal_tag >`

<h3 id="struct-random_access_device_iterator_tag">
<a href="/thrust/api/classes/structrandom__access__device__iterator__tag.html">Struct <code>random&#95;access&#95;device&#95;iterator&#95;tag</code>
</a>
</h3>

**Inherits From**:
`thrust::detail::iterator_category_with_system_and_traversal< std::random_access_iterator_tag, thrust::device_system_tag, thrust::random_access_traversal_tag >`


## Types

<h3 id="typedef-input_host_iterator_tag">
Typedef <code>input&#95;host&#95;iterator&#95;tag</code>
</h3>

<code class="doxybook">
<span>typedef std::input_iterator_tag<b>input_host_iterator_tag</b>;</span></code>
<code>input&#95;host&#95;iterator&#95;tag</code> is an empty class: it has no member functions, member variables, or nested types. It is used solely as a "tag": a representation of the Input Host Iterator concept within the C++ type system.

**See**:
<a href="https://en.cppreference.com/w/cpp/iterator/iterator_tags">https://en.cppreference.com/w/cpp/iterator/iterator_tags</a><a href="/thrust/api/classes/structiterator__traits.html">iterator_traits</a>, <a href="/thrust/api/classes/structinput__device__iterator__tag.html">input_device_iterator_tag</a>, <a href="/thrust/api/classes/structoutput__device__iterator__tag.html">output_device_iterator_tag</a>, <a href="/thrust/api/classes/structforward__device__iterator__tag.html">forward_device_iterator_tag</a>, <a href="/thrust/api/classes/structbidirectional__device__iterator__tag.html">bidirectional_device_iterator_tag</a>, <a href="/thrust/api/classes/structrandom__access__device__iterator__tag.html">random_access_device_iterator_tag</a>, <a href="/thrust/api/groups/group__iterator__tag__classes.html#typedef-output_host_iterator_tag">output_host_iterator_tag</a>, <a href="/thrust/api/groups/group__iterator__tag__classes.html#typedef-forward_host_iterator_tag">forward_host_iterator_tag</a>, <a href="/thrust/api/groups/group__iterator__tag__classes.html#typedef-bidirectional_host_iterator_tag">bidirectional_host_iterator_tag</a>, <a href="/thrust/api/groups/group__iterator__tag__classes.html#typedef-random_access_host_iterator_tag">random_access_host_iterator_tag</a>

<h3 id="typedef-output_host_iterator_tag">
Typedef <code>output&#95;host&#95;iterator&#95;tag</code>
</h3>

<code class="doxybook">
<span>typedef std::output_iterator_tag<b>output_host_iterator_tag</b>;</span></code>
<code>output&#95;host&#95;iterator&#95;tag</code> is an empty class: it has no member functions, member variables, or nested types. It is used solely as a "tag": a representation of the Output Host Iterator concept within the C++ type system.

**See**:
<a href="https://en.cppreference.com/w/cpp/iterator/iterator_tags">https://en.cppreference.com/w/cpp/iterator/iterator_tags</a><a href="/thrust/api/classes/structiterator__traits.html">iterator_traits</a>, <a href="/thrust/api/classes/structinput__device__iterator__tag.html">input_device_iterator_tag</a>, <a href="/thrust/api/classes/structoutput__device__iterator__tag.html">output_device_iterator_tag</a>, <a href="/thrust/api/classes/structforward__device__iterator__tag.html">forward_device_iterator_tag</a>, <a href="/thrust/api/classes/structbidirectional__device__iterator__tag.html">bidirectional_device_iterator_tag</a>, <a href="/thrust/api/classes/structrandom__access__device__iterator__tag.html">random_access_device_iterator_tag</a>, <a href="/thrust/api/groups/group__iterator__tag__classes.html#typedef-input_host_iterator_tag">input_host_iterator_tag</a>, <a href="/thrust/api/groups/group__iterator__tag__classes.html#typedef-forward_host_iterator_tag">forward_host_iterator_tag</a>, <a href="/thrust/api/groups/group__iterator__tag__classes.html#typedef-bidirectional_host_iterator_tag">bidirectional_host_iterator_tag</a>, <a href="/thrust/api/groups/group__iterator__tag__classes.html#typedef-random_access_host_iterator_tag">random_access_host_iterator_tag</a>

<h3 id="typedef-forward_host_iterator_tag">
Typedef <code>forward&#95;host&#95;iterator&#95;tag</code>
</h3>

<code class="doxybook">
<span>typedef std::forward_iterator_tag<b>forward_host_iterator_tag</b>;</span></code>
<code>forward&#95;host&#95;iterator&#95;tag</code> is an empty class: it has no member functions, member variables, or nested types. It is used solely as a "tag": a representation of the Forward Host Iterator concept within the C++ type system.

**See**:
<a href="https://en.cppreference.com/w/cpp/iterator/iterator_tags">https://en.cppreference.com/w/cpp/iterator/iterator_tags</a><a href="/thrust/api/classes/structiterator__traits.html">iterator_traits</a>, <a href="/thrust/api/classes/structinput__device__iterator__tag.html">input_device_iterator_tag</a>, <a href="/thrust/api/classes/structoutput__device__iterator__tag.html">output_device_iterator_tag</a>, <a href="/thrust/api/classes/structforward__device__iterator__tag.html">forward_device_iterator_tag</a>, <a href="/thrust/api/classes/structbidirectional__device__iterator__tag.html">bidirectional_device_iterator_tag</a>, <a href="/thrust/api/classes/structrandom__access__device__iterator__tag.html">random_access_device_iterator_tag</a>, <a href="/thrust/api/groups/group__iterator__tag__classes.html#typedef-input_host_iterator_tag">input_host_iterator_tag</a>, <a href="/thrust/api/groups/group__iterator__tag__classes.html#typedef-output_host_iterator_tag">output_host_iterator_tag</a>, <a href="/thrust/api/groups/group__iterator__tag__classes.html#typedef-bidirectional_host_iterator_tag">bidirectional_host_iterator_tag</a>, <a href="/thrust/api/groups/group__iterator__tag__classes.html#typedef-random_access_host_iterator_tag">random_access_host_iterator_tag</a>

<h3 id="typedef-bidirectional_host_iterator_tag">
Typedef <code>bidirectional&#95;host&#95;iterator&#95;tag</code>
</h3>

<code class="doxybook">
<span>typedef std::bidirectional_iterator_tag<b>bidirectional_host_iterator_tag</b>;</span></code>
<code>bidirectional&#95;host&#95;iterator&#95;tag</code> is an empty class: it has no member functions, member variables, or nested types. It is used solely as a "tag": a representation of the Forward Host Iterator concept within the C++ type system.

**See**:
<a href="https://en.cppreference.com/w/cpp/iterator/iterator_tags">https://en.cppreference.com/w/cpp/iterator/iterator_tags</a><a href="/thrust/api/classes/structiterator__traits.html">iterator_traits</a>, <a href="/thrust/api/classes/structinput__device__iterator__tag.html">input_device_iterator_tag</a>, <a href="/thrust/api/classes/structoutput__device__iterator__tag.html">output_device_iterator_tag</a>, <a href="/thrust/api/classes/structforward__device__iterator__tag.html">forward_device_iterator_tag</a>, <a href="/thrust/api/classes/structbidirectional__device__iterator__tag.html">bidirectional_device_iterator_tag</a>, <a href="/thrust/api/classes/structrandom__access__device__iterator__tag.html">random_access_device_iterator_tag</a>, <a href="/thrust/api/groups/group__iterator__tag__classes.html#typedef-input_host_iterator_tag">input_host_iterator_tag</a>, <a href="/thrust/api/groups/group__iterator__tag__classes.html#typedef-output_host_iterator_tag">output_host_iterator_tag</a>, <a href="/thrust/api/groups/group__iterator__tag__classes.html#typedef-forward_host_iterator_tag">forward_host_iterator_tag</a>, <a href="/thrust/api/groups/group__iterator__tag__classes.html#typedef-random_access_host_iterator_tag">random_access_host_iterator_tag</a>

<h3 id="typedef-random_access_host_iterator_tag">
Typedef <code>random&#95;access&#95;host&#95;iterator&#95;tag</code>
</h3>

<code class="doxybook">
<span>typedef std::random_access_iterator_tag<b>random_access_host_iterator_tag</b>;</span></code>
<code>random&#95;access&#95;host&#95;iterator&#95;tag</code> is an empty class: it has no member functions, member variables, or nested types. It is used solely as a "tag": a representation of the Forward Host Iterator concept within the C++ type system.

**See**:
<a href="https://en.cppreference.com/w/cpp/iterator/iterator_tags">https://en.cppreference.com/w/cpp/iterator/iterator_tags</a><a href="/thrust/api/classes/structiterator__traits.html">iterator_traits</a>, <a href="/thrust/api/classes/structinput__device__iterator__tag.html">input_device_iterator_tag</a>, <a href="/thrust/api/classes/structoutput__device__iterator__tag.html">output_device_iterator_tag</a>, <a href="/thrust/api/classes/structforward__device__iterator__tag.html">forward_device_iterator_tag</a>, <a href="/thrust/api/classes/structbidirectional__device__iterator__tag.html">bidirectional_device_iterator_tag</a>, <a href="/thrust/api/classes/structrandom__access__device__iterator__tag.html">random_access_device_iterator_tag</a>, <a href="/thrust/api/groups/group__iterator__tag__classes.html#typedef-input_host_iterator_tag">input_host_iterator_tag</a>, <a href="/thrust/api/groups/group__iterator__tag__classes.html#typedef-output_host_iterator_tag">output_host_iterator_tag</a>, <a href="/thrust/api/groups/group__iterator__tag__classes.html#typedef-forward_host_iterator_tag">forward_host_iterator_tag</a>, <a href="/thrust/api/groups/group__iterator__tag__classes.html#typedef-bidirectional_host_iterator_tag">bidirectional_host_iterator_tag</a>


