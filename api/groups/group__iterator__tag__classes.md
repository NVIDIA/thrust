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
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1input__device__iterator__tag.html">thrust::input&#95;device&#95;iterator&#95;tag</a></b>;</span>
<br>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1output__device__iterator__tag.html">thrust::output&#95;device&#95;iterator&#95;tag</a></b>;</span>
<br>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1forward__device__iterator__tag.html">thrust::forward&#95;device&#95;iterator&#95;tag</a></b>;</span>
<br>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1bidirectional__device__iterator__tag.html">thrust::bidirectional&#95;device&#95;iterator&#95;tag</a></b>;</span>
<br>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1random__access__device__iterator__tag.html">thrust::random&#95;access&#95;device&#95;iterator&#95;tag</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__iterator__tag__classes.html#typedef-input-host-iterator-tag">thrust::input&#95;host&#95;iterator&#95;tag</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__iterator__tag__classes.html#typedef-output-host-iterator-tag">thrust::output&#95;host&#95;iterator&#95;tag</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__iterator__tag__classes.html#typedef-forward-host-iterator-tag">thrust::forward&#95;host&#95;iterator&#95;tag</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__iterator__tag__classes.html#typedef-bidirectional-host-iterator-tag">thrust::bidirectional&#95;host&#95;iterator&#95;tag</a></b>;</span>
<br>
<span>typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/groups/group__iterator__tag__classes.html#typedef-random-access-host-iterator-tag">thrust::random&#95;access&#95;host&#95;iterator&#95;tag</a></b>;</span>
</code>

## Member Classes

<h3 id="struct-thrustinput-device-iterator-tag">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1input__device__iterator__tag.html">Struct <code>thrust::input&#95;device&#95;iterator&#95;tag</code>
</a>
</h3>

**Inherits From**:
`thrust::detail::iterator_category_with_system_and_traversal< std::input_iterator_tag, thrust::device_system_tag, thrust::single_pass_traversal_tag >`

<h3 id="struct-thrustoutput-device-iterator-tag">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1output__device__iterator__tag.html">Struct <code>thrust::output&#95;device&#95;iterator&#95;tag</code>
</a>
</h3>

**Inherits From**:
`thrust::detail::iterator_category_with_system_and_traversal< std::output_iterator_tag, thrust::device_system_tag, thrust::single_pass_traversal_tag >`

<h3 id="struct-thrustforward-device-iterator-tag">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1forward__device__iterator__tag.html">Struct <code>thrust::forward&#95;device&#95;iterator&#95;tag</code>
</a>
</h3>

**Inherits From**:
`thrust::detail::iterator_category_with_system_and_traversal< std::forward_iterator_tag, thrust::device_system_tag, thrust::forward_traversal_tag >`

<h3 id="struct-thrustbidirectional-device-iterator-tag">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1bidirectional__device__iterator__tag.html">Struct <code>thrust::bidirectional&#95;device&#95;iterator&#95;tag</code>
</a>
</h3>

**Inherits From**:
`thrust::detail::iterator_category_with_system_and_traversal< std::bidirectional_iterator_tag, thrust::device_system_tag, thrust::bidirectional_traversal_tag >`

<h3 id="struct-thrustrandom-access-device-iterator-tag">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1random__access__device__iterator__tag.html">Struct <code>thrust::random&#95;access&#95;device&#95;iterator&#95;tag</code>
</a>
</h3>

**Inherits From**:
`thrust::detail::iterator_category_with_system_and_traversal< std::random_access_iterator_tag, thrust::device_system_tag, thrust::random_access_traversal_tag >`


## Types

<h3 id="typedef-input-host-iterator-tag">
Typedef <code>thrust::input&#95;host&#95;iterator&#95;tag</code>
</h3>

<code class="doxybook">
<span>typedef std::input_iterator_tag<b>input_host_iterator_tag</b>;</span></code>
<code>input&#95;host&#95;iterator&#95;tag</code> is an empty class: it has no member functions, member variables, or nested types. It is used solely as a "tag": a representation of the Input Host Iterator concept within the C++ type system.

**See**:
<a href="https://en.cppreference.com/w/cpp/iterator/iterator_tags">https://en.cppreference.com/w/cpp/iterator/iterator_tags</a><a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">iterator_traits</a>, <a href="{{ site.baseurl }}/api/classes/structthrust_1_1input__device__iterator__tag.html">input_device_iterator_tag</a>, <a href="{{ site.baseurl }}/api/classes/structthrust_1_1output__device__iterator__tag.html">output_device_iterator_tag</a>, <a href="{{ site.baseurl }}/api/classes/structthrust_1_1forward__device__iterator__tag.html">forward_device_iterator_tag</a>, <a href="{{ site.baseurl }}/api/classes/structthrust_1_1bidirectional__device__iterator__tag.html">bidirectional_device_iterator_tag</a>, <a href="{{ site.baseurl }}/api/classes/structthrust_1_1random__access__device__iterator__tag.html">random_access_device_iterator_tag</a>, output_host_iterator_tag, forward_host_iterator_tag, bidirectional_host_iterator_tag, random_access_host_iterator_tag 

<h3 id="typedef-output-host-iterator-tag">
Typedef <code>thrust::output&#95;host&#95;iterator&#95;tag</code>
</h3>

<code class="doxybook">
<span>typedef std::output_iterator_tag<b>output_host_iterator_tag</b>;</span></code>
<code>output&#95;host&#95;iterator&#95;tag</code> is an empty class: it has no member functions, member variables, or nested types. It is used solely as a "tag": a representation of the Output Host Iterator concept within the C++ type system.

**See**:
<a href="https://en.cppreference.com/w/cpp/iterator/iterator_tags">https://en.cppreference.com/w/cpp/iterator/iterator_tags</a><a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">iterator_traits</a>, <a href="{{ site.baseurl }}/api/classes/structthrust_1_1input__device__iterator__tag.html">input_device_iterator_tag</a>, <a href="{{ site.baseurl }}/api/classes/structthrust_1_1output__device__iterator__tag.html">output_device_iterator_tag</a>, <a href="{{ site.baseurl }}/api/classes/structthrust_1_1forward__device__iterator__tag.html">forward_device_iterator_tag</a>, <a href="{{ site.baseurl }}/api/classes/structthrust_1_1bidirectional__device__iterator__tag.html">bidirectional_device_iterator_tag</a>, <a href="{{ site.baseurl }}/api/classes/structthrust_1_1random__access__device__iterator__tag.html">random_access_device_iterator_tag</a>, input_host_iterator_tag, forward_host_iterator_tag, bidirectional_host_iterator_tag, random_access_host_iterator_tag 

<h3 id="typedef-forward-host-iterator-tag">
Typedef <code>thrust::forward&#95;host&#95;iterator&#95;tag</code>
</h3>

<code class="doxybook">
<span>typedef std::forward_iterator_tag<b>forward_host_iterator_tag</b>;</span></code>
<code>forward&#95;host&#95;iterator&#95;tag</code> is an empty class: it has no member functions, member variables, or nested types. It is used solely as a "tag": a representation of the Forward Host Iterator concept within the C++ type system.

**See**:
<a href="https://en.cppreference.com/w/cpp/iterator/iterator_tags">https://en.cppreference.com/w/cpp/iterator/iterator_tags</a><a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">iterator_traits</a>, <a href="{{ site.baseurl }}/api/classes/structthrust_1_1input__device__iterator__tag.html">input_device_iterator_tag</a>, <a href="{{ site.baseurl }}/api/classes/structthrust_1_1output__device__iterator__tag.html">output_device_iterator_tag</a>, <a href="{{ site.baseurl }}/api/classes/structthrust_1_1forward__device__iterator__tag.html">forward_device_iterator_tag</a>, <a href="{{ site.baseurl }}/api/classes/structthrust_1_1bidirectional__device__iterator__tag.html">bidirectional_device_iterator_tag</a>, <a href="{{ site.baseurl }}/api/classes/structthrust_1_1random__access__device__iterator__tag.html">random_access_device_iterator_tag</a>, input_host_iterator_tag, output_host_iterator_tag, bidirectional_host_iterator_tag, random_access_host_iterator_tag 

<h3 id="typedef-bidirectional-host-iterator-tag">
Typedef <code>thrust::bidirectional&#95;host&#95;iterator&#95;tag</code>
</h3>

<code class="doxybook">
<span>typedef std::bidirectional_iterator_tag<b>bidirectional_host_iterator_tag</b>;</span></code>
<code>bidirectional&#95;host&#95;iterator&#95;tag</code> is an empty class: it has no member functions, member variables, or nested types. It is used solely as a "tag": a representation of the Forward Host Iterator concept within the C++ type system.

**See**:
<a href="https://en.cppreference.com/w/cpp/iterator/iterator_tags">https://en.cppreference.com/w/cpp/iterator/iterator_tags</a><a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">iterator_traits</a>, <a href="{{ site.baseurl }}/api/classes/structthrust_1_1input__device__iterator__tag.html">input_device_iterator_tag</a>, <a href="{{ site.baseurl }}/api/classes/structthrust_1_1output__device__iterator__tag.html">output_device_iterator_tag</a>, <a href="{{ site.baseurl }}/api/classes/structthrust_1_1forward__device__iterator__tag.html">forward_device_iterator_tag</a>, <a href="{{ site.baseurl }}/api/classes/structthrust_1_1bidirectional__device__iterator__tag.html">bidirectional_device_iterator_tag</a>, <a href="{{ site.baseurl }}/api/classes/structthrust_1_1random__access__device__iterator__tag.html">random_access_device_iterator_tag</a>, input_host_iterator_tag, output_host_iterator_tag, forward_host_iterator_tag, random_access_host_iterator_tag 

<h3 id="typedef-random-access-host-iterator-tag">
Typedef <code>thrust::random&#95;access&#95;host&#95;iterator&#95;tag</code>
</h3>

<code class="doxybook">
<span>typedef std::random_access_iterator_tag<b>random_access_host_iterator_tag</b>;</span></code>
<code>random&#95;access&#95;host&#95;iterator&#95;tag</code> is an empty class: it has no member functions, member variables, or nested types. It is used solely as a "tag": a representation of the Forward Host Iterator concept within the C++ type system.

**See**:
<a href="https://en.cppreference.com/w/cpp/iterator/iterator_tags">https://en.cppreference.com/w/cpp/iterator/iterator_tags</a><a href="{{ site.baseurl }}/api/classes/structthrust_1_1iterator__traits.html">iterator_traits</a>, <a href="{{ site.baseurl }}/api/classes/structthrust_1_1input__device__iterator__tag.html">input_device_iterator_tag</a>, <a href="{{ site.baseurl }}/api/classes/structthrust_1_1output__device__iterator__tag.html">output_device_iterator_tag</a>, <a href="{{ site.baseurl }}/api/classes/structthrust_1_1forward__device__iterator__tag.html">forward_device_iterator_tag</a>, <a href="{{ site.baseurl }}/api/classes/structthrust_1_1bidirectional__device__iterator__tag.html">bidirectional_device_iterator_tag</a>, <a href="{{ site.baseurl }}/api/classes/structthrust_1_1random__access__device__iterator__tag.html">random_access_device_iterator_tag</a>, input_host_iterator_tag, output_host_iterator_tag, forward_host_iterator_tag, bidirectional_host_iterator_tag 


