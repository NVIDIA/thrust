---
title: thrust::iterator_adaptor
parent: Fancy Iterators
grand_parent: Iterators
nav_exclude: true
has_children: true
has_toc: false
---

# Class `thrust::iterator_adaptor`

<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__adaptor.html">iterator&#95;adaptor</a></code> is an iterator which adapts an existing type of iterator to create a new type of iterator. Most of Thrust's fancy iterators are defined via inheritance from <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__adaptor.html">iterator&#95;adaptor</a></code>. While composition of these existing Thrust iterators is often sufficient for expressing the desired functionality, it is occasionally more straightforward to derive from <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__adaptor.html">iterator&#95;adaptor</a></code> directly.

To see how to use <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__adaptor.html">iterator&#95;adaptor</a></code> to create a novel iterator type, let's examine how to use it to define <code>repeat&#95;iterator</code>, a fancy iterator which repeats elements from another range a given number of time:



```cpp
#include <thrust/iterator/iterator_adaptor.h>

// derive repeat_iterator from iterator_adaptor
template<typename Iterator>
  class repeat_iterator
    : public thrust::iterator_adaptor<
        repeat_iterator<Iterator>, // the first template parameter is the name of the iterator we're creating
        Iterator                   // the second template parameter is the name of the iterator we're adapting
                                   // we can use the default for the additional template parameters
      >
{
  public:
    // shorthand for the name of the iterator_adaptor we're deriving from
    typedef thrust::iterator_adaptor<
      repeat_iterator<Iterator>,
      Iterator
    > super_t;

    __host__ __device__
    repeat_iterator(const Iterator &x, int n) : super_t(x), begin(x), n(n) {}

    // befriend thrust::iterator_core_access to allow it access to the private interface below
    friend class thrust::iterator_core_access;

  private:
    // repeat each element of the adapted range n times
    unsigned int n;

    // used to keep track of where we began
    const Iterator begin;

    // it is private because only thrust::iterator_core_access needs access to it
    __host__ __device__
    typename super_t::reference dereference() const
    {
      return *(begin + (this->base() - begin) / n);
    }
};
```

Except for the first two, <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__adaptor.html">iterator&#95;adaptor</a>'s</code> template parameters are optional. When omitted, or when the user specifies <code>thrust::use&#95;default</code> in its place, <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__adaptor.html">iterator&#95;adaptor</a></code> will use a default type inferred from <code>Base</code>.

<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__adaptor.html">iterator&#95;adaptor</a>'s</code> functionality is derived from and generally equivalent to <code>boost::iterator&#95;adaptor</code>. The exception is Thrust's addition of the template parameter <code>System</code>, which is necessary to allow Thrust to dispatch an algorithm to one of several parallel backend systems.

<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__adaptor.html">iterator&#95;adaptor</a></code> is a powerful tool for creating custom iterators directly. However, the large set of iterator semantics which must be satisfied for algorithm compatibility can make <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__adaptor.html">iterator&#95;adaptor</a></code> difficult to use correctly. Unless you require the full expressivity of <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__adaptor.html">iterator&#95;adaptor</a></code>, consider building a custom iterator through composition of existing higher-level fancy iterators instead.

Interested users may refer to <code>boost::iterator&#95;adaptor</code>'s documentation for further usage examples. 

**Inherits From**:
`detail::iterator_adaptor_base::type`

<code class="doxybook">
<span>#include <thrust/iterator/iterator_adaptor.h></span><br>
<span>template &lt;typename Derived,</span>
<span>&nbsp;&nbsp;typename Base,</span>
<span>&nbsp;&nbsp;typename Value = use&#95;default,</span>
<span>&nbsp;&nbsp;typename System = use&#95;default,</span>
<span>&nbsp;&nbsp;typename Traversal = use&#95;default,</span>
<span>&nbsp;&nbsp;typename Reference = use&#95;default,</span>
<span>&nbsp;&nbsp;typename Difference = use&#95;default&gt;</span>
<span>class thrust::iterator&#95;adaptor {</span>
<span>public:</span><span>&nbsp;&nbsp;typedef <i>see below</i> <b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__adaptor.html#typedef-base-type">base&#95;type</a></b>;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__adaptor.html#function-iterator-adaptor">iterator&#95;adaptor</a></b>();</span>
<br>
<span>&nbsp;&nbsp;explicit __thrust_exec_check_disable__ __host__ __device__ </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__adaptor.html#function-iterator-adaptor">iterator&#95;adaptor</a></b>(Base const & iter);</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ Base const & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__adaptor.html#function-base">base</a></b>() const;</span>
<br>
<span>protected:</span><span>&nbsp;&nbsp;__host__ __device__ Base const & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__adaptor.html#function-base-reference">base&#95;reference</a></b>() const;</span>
<br>
<span>&nbsp;&nbsp;__host__ __device__ Base & </span><span>&nbsp;&nbsp;<b><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__adaptor.html#function-base-reference">base&#95;reference</a></b>();</span>
<span>};</span>
</code>

## Member Types

<h3 id="typedef-base-type">
Typedef <code>thrust::iterator&#95;adaptor::base&#95;type</code>
</h3>

<code class="doxybook">
<span>typedef Base<b>base_type</b>;</span></code>
The type of iterator this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__adaptor.html">iterator&#95;adaptor</a>'s</code><code>adapts</code>. 


## Member Functions

<h3 id="function-iterator-adaptor">
Function <code>thrust::iterator&#95;adaptor::iterator&#95;adaptor</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ </span><span><b>iterator_adaptor</b>();</span></code>
<code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__adaptor.html">iterator&#95;adaptor</a>'s</code> default constructor does nothing. 

<h3 id="function-iterator-adaptor">
Function <code>thrust::iterator&#95;adaptor::iterator&#95;adaptor</code>
</h3>

<code class="doxybook">
<span>explicit __thrust_exec_check_disable__ __host__ __device__ </span><span><b>iterator_adaptor</b>(Base const & iter);</span></code>
This constructor copies from a given instance of the <code>Base</code> iterator. 

<h3 id="function-base">
Function <code>thrust::iterator&#95;adaptor::base</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ Base const & </span><span><b>base</b>() const;</span></code>
**Returns**:
A <code>const</code> reference to the <code>Base</code> iterator this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__adaptor.html">iterator&#95;adaptor</a></code> adapts. 


## Protected Member Functions

<h3 id="function-base-reference">
Function <code>thrust::iterator&#95;adaptor::base&#95;reference</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ Base const & </span><span><b>base_reference</b>() const;</span></code>
**Returns**:
A <code>const</code> reference to the <code>Base</code> iterator this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__adaptor.html">iterator&#95;adaptor</a></code> adapts. 

<h3 id="function-base-reference">
Function <code>thrust::iterator&#95;adaptor::base&#95;reference</code>
</h3>

<code class="doxybook">
<span>__host__ __device__ Base & </span><span><b>base_reference</b>();</span></code>
**Returns**:
A mutable reference to the <code>Base</code> iterator this <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1iterator__adaptor.html">iterator&#95;adaptor</a></code> adapts. 


