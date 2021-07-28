---
title: Fancy Iterators
parent: Iterators
grand_parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Fancy Iterators

<code class="doxybook">
<span>template &lt;typename Value,</span>
<span>&nbsp;&nbsp;typename Incrementable = use&#95;default,</span>
<span>&nbsp;&nbsp;typename System = use&#95;default&gt;</span>
<span>class <b><a href="/api/classes/classconstant__iterator.html">constant&#95;iterator</a></b>;</span>
<br>
<span>template &lt;typename Incrementable,</span>
<span>&nbsp;&nbsp;typename System = use&#95;default,</span>
<span>&nbsp;&nbsp;typename Traversal = use&#95;default,</span>
<span>&nbsp;&nbsp;typename Difference = use&#95;default&gt;</span>
<span>class <b><a href="/api/classes/classcounting__iterator.html">counting&#95;iterator</a></b>;</span>
<br>
<span>template &lt;typename System = use&#95;default&gt;</span>
<span>class <b><a href="/api/classes/classdiscard__iterator.html">discard&#95;iterator</a></b>;</span>
<br>
<span>template &lt;typename Derived,</span>
<span>&nbsp;&nbsp;typename Base,</span>
<span>&nbsp;&nbsp;typename Value = use&#95;default,</span>
<span>&nbsp;&nbsp;typename System = use&#95;default,</span>
<span>&nbsp;&nbsp;typename Traversal = use&#95;default,</span>
<span>&nbsp;&nbsp;typename Reference = use&#95;default,</span>
<span>&nbsp;&nbsp;typename Difference = use&#95;default&gt;</span>
<span>class <b><a href="/api/classes/classiterator__adaptor.html">iterator&#95;adaptor</a></b>;</span>
<br>
<span>template &lt;typename Derived,</span>
<span>&nbsp;&nbsp;typename Value,</span>
<span>&nbsp;&nbsp;typename System,</span>
<span>&nbsp;&nbsp;typename Traversal,</span>
<span>&nbsp;&nbsp;typename Reference,</span>
<span>&nbsp;&nbsp;typename Difference = std::ptrdiff&#95;t&gt;</span>
<span>class <b><a href="/api/classes/classiterator__facade.html">iterator&#95;facade</a></b>;</span>
<br>
<span>class <b><a href="/api/classes/classiterator__core__access.html">iterator&#95;core&#95;access</a></b>;</span>
<br>
<span>template &lt;typename ElementIterator,</span>
<span>&nbsp;&nbsp;typename IndexIterator&gt;</span>
<span>class <b><a href="/api/classes/classpermutation__iterator.html">permutation&#95;iterator</a></b>;</span>
<br>
<span>template &lt;typename BidirectionalIterator&gt;</span>
<span>class <b><a href="/api/classes/classreverse__iterator.html">reverse&#95;iterator</a></b>;</span>
<br>
<span>template &lt;typename InputFunction,</span>
<span>&nbsp;&nbsp;typename OutputFunction,</span>
<span>&nbsp;&nbsp;typename Iterator&gt;</span>
<span>class <b><a href="/api/classes/classtransform__input__output__iterator.html">transform&#95;input&#95;output&#95;iterator</a></b>;</span>
<br>
<span>template &lt;class AdaptableUnaryFunction,</span>
<span>&nbsp;&nbsp;class Iterator,</span>
<span>&nbsp;&nbsp;class Reference = use&#95;default,</span>
<span>&nbsp;&nbsp;class Value = use&#95;default&gt;</span>
<span>class <b><a href="/api/classes/classtransform__iterator.html">transform&#95;iterator</a></b>;</span>
<br>
<span>template &lt;typename UnaryFunction,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span>class <b><a href="/api/classes/classtransform__output__iterator.html">transform&#95;output&#95;iterator</a></b>;</span>
<br>
<span>template &lt;typename IteratorTuple&gt;</span>
<span>class <b><a href="/api/classes/classzip__iterator.html">zip&#95;iterator</a></b>;</span>
<br>
<span>template &lt;typename ValueT,</span>
<span>&nbsp;&nbsp;typename IndexT&gt;</span>
<span>__host__ __device__ <a href="/api/classes/classconstant__iterator.html">constant_iterator</a>< ValueT, IndexT > </span><span><b><a href="/api/groups/group__fancyiterator.html#function-make_constant_iterator">make&#95;constant&#95;iterator</a></b>(ValueT x,</span>
<span>&nbsp;&nbsp;IndexT i = int());</span>
<br>
<span>template &lt;typename V&gt;</span>
<span>__host__ __device__ <a href="/api/classes/classconstant__iterator.html">constant_iterator</a>< V > </span><span><b><a href="/api/groups/group__fancyiterator.html#function-make_constant_iterator">make&#95;constant&#95;iterator</a></b>(V x);</span>
<br>
<span>template &lt;typename Incrementable&gt;</span>
<span>__host__ __device__ <a href="/api/classes/classcounting__iterator.html">counting_iterator</a>< Incrementable > </span><span><b><a href="/api/groups/group__fancyiterator.html#function-make_counting_iterator">make&#95;counting&#95;iterator</a></b>(Incrementable x);</span>
<br>
<span>__host__ __device__ <a href="/api/classes/classdiscard__iterator.html">discard_iterator</a> </span><span><b><a href="/api/groups/group__fancyiterator.html#function-make_discard_iterator">make&#95;discard&#95;iterator</a></b>(<a href="/api/classes/classdiscard__iterator.html">discard_iterator</a><>::difference_type i = discard&#95;iterator&lt;&gt;::difference&#95;type(0));</span>
<br>
<span>template &lt;typename ElementIterator,</span>
<span>&nbsp;&nbsp;typename IndexIterator&gt;</span>
<span>__host__ __device__ <a href="/api/classes/classpermutation__iterator.html">permutation_iterator</a>< ElementIterator, IndexIterator > </span><span><b><a href="/api/groups/group__fancyiterator.html#function-make_permutation_iterator">make&#95;permutation&#95;iterator</a></b>(ElementIterator e,</span>
<span>&nbsp;&nbsp;IndexIterator i);</span>
<br>
<span>template &lt;typename BidirectionalIterator&gt;</span>
<span>__host__ __device__ <a href="/api/classes/classreverse__iterator.html">reverse_iterator</a>< BidirectionalIterator > </span><span><b><a href="/api/groups/group__fancyiterator.html#function-make_reverse_iterator">make&#95;reverse&#95;iterator</a></b>(BidirectionalIterator x);</span>
<br>
<span>template &lt;typename InputFunction,</span>
<span>&nbsp;&nbsp;typename OutputFunction,</span>
<span>&nbsp;&nbsp;typename Iterator&gt;</span>
<span><a href="/api/classes/classtransform__input__output__iterator.html">transform_input_output_iterator</a>< InputFunction, OutputFunction, Iterator > __host__ __device__ </span><span><b><a href="/api/groups/group__fancyiterator.html#function-make_transform_input_output_iterator">make&#95;transform&#95;input&#95;output&#95;iterator</a></b>(Iterator io,</span>
<span>&nbsp;&nbsp;InputFunction input_function,</span>
<span>&nbsp;&nbsp;OutputFunction output_function);</span>
<br>
<span>template &lt;class AdaptableUnaryFunction,</span>
<span>&nbsp;&nbsp;class Iterator&gt;</span>
<span>__host__ __device__ <a href="/api/classes/classtransform__iterator.html">transform_iterator</a>< AdaptableUnaryFunction, Iterator > </span><span><b><a href="/api/groups/group__fancyiterator.html#function-make_transform_iterator">make&#95;transform&#95;iterator</a></b>(Iterator it,</span>
<span>&nbsp;&nbsp;AdaptableUnaryFunction fun);</span>
<br>
<span>template &lt;typename UnaryFunction,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span><a href="/api/classes/classtransform__output__iterator.html">transform_output_iterator</a>< UnaryFunction, OutputIterator > __host__ __device__ </span><span><b><a href="/api/groups/group__fancyiterator.html#function-make_transform_output_iterator">make&#95;transform&#95;output&#95;iterator</a></b>(OutputIterator out,</span>
<span>&nbsp;&nbsp;UnaryFunction fun);</span>
<br>
<span>template &lt;typename... Iterators&gt;</span>
<span>__host__ __device__ <a href="/api/classes/classzip__iterator.html">zip_iterator</a>< thrust::tuple< Iterators... > > </span><span><b><a href="/api/groups/group__fancyiterator.html#function-make_zip_iterator">make&#95;zip&#95;iterator</a></b>(thrust::tuple< Iterators... > t);</span>
<br>
<span>template &lt;typename... Iterators&gt;</span>
<span>__host__ __device__ <a href="/api/classes/classzip__iterator.html">zip_iterator</a>< thrust::tuple< Iterators... > > </span><span><b><a href="/api/groups/group__fancyiterator.html#function-make_zip_iterator">make&#95;zip&#95;iterator</a></b>(Iterators... its);</span>
</code>

## Member Classes

<h3 id="class-constant_iterator">
<a href="/api/classes/classconstant__iterator.html">Class <code>constant&#95;iterator</code>
</a>
</h3>

**Inherits From**:
`detail::constant_iterator_base::type< Value, use_default, use_default >`

<h3 id="class-counting_iterator">
<a href="/api/classes/classcounting__iterator.html">Class <code>counting&#95;iterator</code>
</a>
</h3>

**Inherits From**:
`detail::counting_iterator_base::type< Incrementable, use_default, use_default, use_default >`

<h3 id="class-discard_iterator">
<a href="/api/classes/classdiscard__iterator.html">Class <code>discard&#95;iterator</code>
</a>
</h3>

**Inherits From**:
`detail::discard_iterator_base::type< use_default >`

<h3 id="class-iterator_adaptor">
<a href="/api/classes/classiterator__adaptor.html">Class <code>iterator&#95;adaptor</code>
</a>
</h3>

**Inherits From**:
`detail::iterator_adaptor_base::type< Derived, Base, use_default, use_default, use_default, use_default, use_default >`

<h3 id="class-iterator_facade">
<a href="/api/classes/classiterator__facade.html">Class <code>iterator&#95;facade</code>
</a>
</h3>

<h3 id="class-iterator_core_access">
<a href="/api/classes/classiterator__core__access.html">Class <code>iterator&#95;core&#95;access</code>
</a>
</h3>

<h3 id="class-permutation_iterator">
<a href="/api/classes/classpermutation__iterator.html">Class <code>permutation&#95;iterator</code>
</a>
</h3>

**Inherits From**:
`thrust::detail::permutation_iterator_base::type< ElementIterator, IndexIterator >`

<h3 id="class-reverse_iterator">
<a href="/api/classes/classreverse__iterator.html">Class <code>reverse&#95;iterator</code>
</a>
</h3>

**Inherits From**:
`detail::reverse_iterator_base::type< BidirectionalIterator >`

<h3 id="class-transform_input_output_iterator">
<a href="/api/classes/classtransform__input__output__iterator.html">Class <code>transform&#95;input&#95;output&#95;iterator</code>
</a>
</h3>

**Inherits From**:
`detail::transform_input_output_iterator_base::type< InputFunction, OutputFunction, Iterator >`

<h3 id="class-transform_iterator">
<a href="/api/classes/classtransform__iterator.html">Class <code>transform&#95;iterator</code>
</a>
</h3>

**Inherits From**:
`detail::transform_iterator_base::type< AdaptableUnaryFunction, Iterator, use_default, use_default >`

<h3 id="class-transform_output_iterator">
<a href="/api/classes/classtransform__output__iterator.html">Class <code>transform&#95;output&#95;iterator</code>
</a>
</h3>

**Inherits From**:
`detail::transform_output_iterator_base::type< UnaryFunction, OutputIterator >`

<h3 id="class-zip_iterator">
<a href="/api/classes/classzip__iterator.html">Class <code>zip&#95;iterator</code>
</a>
</h3>

**Inherits From**:
`detail::zip_iterator_base::type< IteratorTuple >`


## Functions

<h3 id="function-make_constant_iterator">
Function <code>make&#95;constant&#95;iterator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ValueT,</span>
<span>&nbsp;&nbsp;typename IndexT&gt;</span>
<span>__host__ __device__ <a href="/api/classes/classconstant__iterator.html">constant_iterator</a>< ValueT, IndexT > </span><span><b>make_constant_iterator</b>(ValueT x,</span>
<span>&nbsp;&nbsp;IndexT i = int());</span></code>
This version of <code>make&#95;constant&#95;iterator</code> creates a <code><a href="/api/classes/classconstant__iterator.html">constant&#95;iterator</a></code> from values given for both value and index. The type of <code><a href="/api/classes/classconstant__iterator.html">constant&#95;iterator</a></code> may be inferred by the compiler from the types of its parameters.

**Function Parameters**:
* **`x`** The value of the returned <code><a href="/api/classes/classconstant__iterator.html">constant&#95;iterator</a>'s</code> constant value. 
* **`i`** The index of the returned <code><a href="/api/classes/classconstant__iterator.html">constant&#95;iterator</a></code> within a sequence. The type of this parameter defaults to <code>int</code>. In the default case, the value of this parameter is <code>0</code>.

**Returns**:
A new <code><a href="/api/classes/classconstant__iterator.html">constant&#95;iterator</a></code> with constant value & index as given by <code>x</code> & <code>i</code>.

**See**:
<a href="/api/classes/classconstant__iterator.html">constant_iterator</a>

<h3 id="function-make_constant_iterator">
Function <code>make&#95;constant&#95;iterator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename V&gt;</span>
<span>__host__ __device__ <a href="/api/classes/classconstant__iterator.html">constant_iterator</a>< V > </span><span><b>make_constant_iterator</b>(V x);</span></code>
This version of <code>make&#95;constant&#95;iterator</code> creates a <code><a href="/api/classes/classconstant__iterator.html">constant&#95;iterator</a></code> using only a parameter for the desired constant value. The value of the returned <code><a href="/api/classes/classconstant__iterator.html">constant&#95;iterator</a>'s</code> index is set to <code>0</code>.

**Function Parameters**:
**`x`**: The value of the returned <code><a href="/api/classes/classconstant__iterator.html">constant&#95;iterator</a>'s</code> constant value. 

**Returns**:
A new <code><a href="/api/classes/classconstant__iterator.html">constant&#95;iterator</a></code> with constant value equal to <code>x</code> and index equal to <code>0</code>. 

**See**:
<a href="/api/classes/classconstant__iterator.html">constant_iterator</a>

<h3 id="function-make_counting_iterator">
Function <code>make&#95;counting&#95;iterator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename Incrementable&gt;</span>
<span>__host__ __device__ <a href="/api/classes/classcounting__iterator.html">counting_iterator</a>< Incrementable > </span><span><b>make_counting_iterator</b>(Incrementable x);</span></code>
<code>make&#95;counting&#95;iterator</code> creates a <code><a href="/api/classes/classcounting__iterator.html">counting&#95;iterator</a></code> using an initial value for its <code>Incrementable</code> counter.

**Function Parameters**:
**`x`**: The initial value of the new <code><a href="/api/classes/classcounting__iterator.html">counting&#95;iterator</a>'s</code> counter. 

**Returns**:
A new <code><a href="/api/classes/classcounting__iterator.html">counting&#95;iterator</a></code> whose counter has been initialized to <code>x</code>. 

<h3 id="function-make_discard_iterator">
Function <code>make&#95;discard&#95;iterator</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ <a href="/api/classes/classdiscard__iterator.html">discard_iterator</a> </span><span><b>make_discard_iterator</b>(<a href="/api/classes/classdiscard__iterator.html">discard_iterator</a><>::difference_type i = discard&#95;iterator&lt;&gt;::difference&#95;type(0));</span></code>
<code>make&#95;discard&#95;iterator</code> creates a <code><a href="/api/classes/classdiscard__iterator.html">discard&#95;iterator</a></code> from an optional index parameter.

**Function Parameters**:
**`i`**: The index of the returned <code><a href="/api/classes/classdiscard__iterator.html">discard&#95;iterator</a></code> within a range. In the default case, the value of this parameter is <code>0</code>.

**Returns**:
A new <code><a href="/api/classes/classdiscard__iterator.html">discard&#95;iterator</a></code> with index as given by <code>i</code>.

**See**:
<a href="/api/classes/classconstant__iterator.html">constant_iterator</a>

<h3 id="function-make_permutation_iterator">
Function <code>make&#95;permutation&#95;iterator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename ElementIterator,</span>
<span>&nbsp;&nbsp;typename IndexIterator&gt;</span>
<span>__host__ __device__ <a href="/api/classes/classpermutation__iterator.html">permutation_iterator</a>< ElementIterator, IndexIterator > </span><span><b>make_permutation_iterator</b>(ElementIterator e,</span>
<span>&nbsp;&nbsp;IndexIterator i);</span></code>
<code>make&#95;permutation&#95;iterator</code> creates a <code><a href="/api/classes/classpermutation__iterator.html">permutation&#95;iterator</a></code> from an <code>ElementIterator</code> pointing to a range of elements to "permute" and an <code>IndexIterator</code> pointing to a range of indices defining an indexing scheme on the values.

**Function Parameters**:
* **`e`** An <code>ElementIterator</code> pointing to a range of values. 
* **`i`** An <code>IndexIterator</code> pointing to an indexing scheme to use on <code>e</code>. 

**Returns**:
A new <code><a href="/api/classes/classpermutation__iterator.html">permutation&#95;iterator</a></code> which permutes the range <code>e</code> by <code>i</code>. 

**See**:
<a href="/api/classes/classpermutation__iterator.html">permutation_iterator</a>

<h3 id="function-make_reverse_iterator">
Function <code>make&#95;reverse&#95;iterator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename BidirectionalIterator&gt;</span>
<span>__host__ __device__ <a href="/api/classes/classreverse__iterator.html">reverse_iterator</a>< BidirectionalIterator > </span><span><b>make_reverse_iterator</b>(BidirectionalIterator x);</span></code>
<code>make&#95;reverse&#95;iterator</code> creates a <code><a href="/api/classes/classreverse__iterator.html">reverse&#95;iterator</a></code> from a <code>BidirectionalIterator</code> pointing to a range of elements to reverse.

**Function Parameters**:
**`x`**: A <code>BidirectionalIterator</code> pointing to a range to reverse. 

**Returns**:
A new <code><a href="/api/classes/classreverse__iterator.html">reverse&#95;iterator</a></code> which reverses the range <code>x</code>. 

<h3 id="function-make_transform_input_output_iterator">
Function <code>make&#95;transform&#95;input&#95;output&#95;iterator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename InputFunction,</span>
<span>&nbsp;&nbsp;typename OutputFunction,</span>
<span>&nbsp;&nbsp;typename Iterator&gt;</span>
<span><a href="/api/classes/classtransform__input__output__iterator.html">transform_input_output_iterator</a>< InputFunction, OutputFunction, Iterator > __host__ __device__ </span><span><b>make_transform_input_output_iterator</b>(Iterator io,</span>
<span>&nbsp;&nbsp;InputFunction input_function,</span>
<span>&nbsp;&nbsp;OutputFunction output_function);</span></code>
<code>make&#95;transform&#95;input&#95;output&#95;iterator</code> creates a <code><a href="/api/classes/classtransform__input__output__iterator.html">transform&#95;input&#95;output&#95;iterator</a></code> from an <code>Iterator</code> a <code>InputFunction</code> and a <code>OutputFunction</code>

**Function Parameters**:
* **`io`** An <code>Iterator</code> pointing to where the input to <code>InputFunction</code> will be read from and the result of <code>OutputFunction</code> will be written to 
* **`input_function`** An <code>InputFunction</code> to be executed on values read from the iterator 
* **`output_function`** An <code>OutputFunction</code> to be executed on values written to the iterator 

**See**:
<a href="/api/classes/classtransform__input__output__iterator.html">transform_input_output_iterator</a>

<h3 id="function-make_transform_iterator">
Function <code>make&#95;transform&#95;iterator</code>
</h3>

<code class="doxybook">
<span>template &lt;class AdaptableUnaryFunction,</span>
<span>&nbsp;&nbsp;class Iterator&gt;</span>
<span>__host__ __device__ <a href="/api/classes/classtransform__iterator.html">transform_iterator</a>< AdaptableUnaryFunction, Iterator > </span><span><b>make_transform_iterator</b>(Iterator it,</span>
<span>&nbsp;&nbsp;AdaptableUnaryFunction fun);</span></code>
<code>make&#95;transform&#95;iterator</code> creates a <code><a href="/api/classes/classtransform__iterator.html">transform&#95;iterator</a></code> from an <code>Iterator</code> and <code>AdaptableUnaryFunction</code>.

**Function Parameters**:
* **`it`** The <code>Iterator</code> pointing to the input range of the newly created <code><a href="/api/classes/classtransform__iterator.html">transform&#95;iterator</a></code>. 
* **`fun`** The <code>AdaptableUnaryFunction</code> used to transform the range pointed to by <code>it</code> in the newly created <code><a href="/api/classes/classtransform__iterator.html">transform&#95;iterator</a></code>. 

**Returns**:
A new <code><a href="/api/classes/classtransform__iterator.html">transform&#95;iterator</a></code> which transforms the range at <code>it</code> by <code>fun</code>. 

**See**:
<a href="/api/classes/classtransform__iterator.html">transform_iterator</a>

<h3 id="function-make_transform_output_iterator">
Function <code>make&#95;transform&#95;output&#95;iterator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename UnaryFunction,</span>
<span>&nbsp;&nbsp;typename OutputIterator&gt;</span>
<span><a href="/api/classes/classtransform__output__iterator.html">transform_output_iterator</a>< UnaryFunction, OutputIterator > __host__ __device__ </span><span><b>make_transform_output_iterator</b>(OutputIterator out,</span>
<span>&nbsp;&nbsp;UnaryFunction fun);</span></code>
<code>make&#95;transform&#95;output&#95;iterator</code> creates a <code><a href="/api/classes/classtransform__output__iterator.html">transform&#95;output&#95;iterator</a></code> from an <code>OutputIterator</code> and <code>UnaryFunction</code>.

**Function Parameters**:
* **`out`** The <code>OutputIterator</code> pointing to the output range of the newly created <code><a href="/api/classes/classtransform__output__iterator.html">transform&#95;output&#95;iterator</a></code>
* **`fun`** The <code>UnaryFunction</code> transform the object before assigning it to <code>out</code> by the newly created <code><a href="/api/classes/classtransform__output__iterator.html">transform&#95;output&#95;iterator</a></code>

**See**:
<a href="/api/classes/classtransform__output__iterator.html">transform_output_iterator</a>

<h3 id="function-make_zip_iterator">
Function <code>make&#95;zip&#95;iterator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename... Iterators&gt;</span>
<span>__host__ __device__ <a href="/api/classes/classzip__iterator.html">zip_iterator</a>< thrust::tuple< Iterators... > > </span><span><b>make_zip_iterator</b>(thrust::tuple< Iterators... > t);</span></code>
<code>make&#95;zip&#95;iterator</code> creates a <code><a href="/api/classes/classzip__iterator.html">zip&#95;iterator</a></code> from a <code>tuple</code> of iterators.

**Function Parameters**:
**`t`**: The <code>tuple</code> of iterators to copy. 

**Returns**:
A newly created <code><a href="/api/classes/classzip__iterator.html">zip&#95;iterator</a></code> which zips the iterators encapsulated in <code>t</code>.

**See**:
<a href="/api/classes/classzip__iterator.html">zip_iterator</a>

<h3 id="function-make_zip_iterator">
Function <code>make&#95;zip&#95;iterator</code>
</h3>

<code class="doxybook">
<span>template &lt;typename... Iterators&gt;</span>
<span>__host__ __device__ <a href="/api/classes/classzip__iterator.html">zip_iterator</a>< thrust::tuple< Iterators... > > </span><span><b>make_zip_iterator</b>(Iterators... its);</span></code>
<code>make&#95;zip&#95;iterator</code> creates a <code><a href="/api/classes/classzip__iterator.html">zip&#95;iterator</a></code> from iterators.

**Function Parameters**:
**`its`**: The iterators to copy. 

**Returns**:
A newly created <code><a href="/api/classes/classzip__iterator.html">zip&#95;iterator</a></code> which zips the iterators.

**See**:
<a href="/api/classes/classzip__iterator.html">zip_iterator</a>


