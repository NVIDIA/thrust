---
title: Parallel Execution Policies
parent: API
nav_exclude: false
has_children: true
has_toc: false
---

# Parallel Execution Policies

<code class="doxybook">
<span>template &lt;typename DerivedPolicy&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1host__execution__policy.html">thrust::host&#95;execution&#95;policy</a></b>;</span>
<br>
<span>template &lt;typename DerivedPolicy&gt;</span>
<span>struct <b><a href="{{ site.baseurl }}/api/classes/structthrust_1_1device__execution__policy.html">thrust::device&#95;execution&#95;policy</a></b>;</span>
<br>
<span>static const detail::host_t <b><a href="{{ site.baseurl }}/api/groups/group__execution__policies.html#variable-host">thrust::host</a></b>;</span>
<br>
<span>THRUST_INLINE_CONSTANT detail::device_t <b><a href="{{ site.baseurl }}/api/groups/group__execution__policies.html#variable-device">thrust::device</a></b>;</span>
</code>

## Member Classes

<h3 id="struct-thrusthost-execution-policy">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1host__execution__policy.html">Struct <code>thrust::host&#95;execution&#95;policy</code>
</a>
</h3>

**Inherits From**:
`thrust::system::__THRUST_HOST_SYSTEM_NAMESPACE::execution_policy< DerivedPolicy >`

<h3 id="struct-thrustdevice-execution-policy">
<a href="{{ site.baseurl }}/api/classes/structthrust_1_1device__execution__policy.html">Struct <code>thrust::device&#95;execution&#95;policy</code>
</a>
</h3>

**Inherits From**:
`thrust::system::__THRUST_DEVICE_SYSTEM_NAMESPACE::execution_policy< DerivedPolicy >`


## Variables

<h3 id="variable-host">
Variable <code>thrust::host</code>
</h3>

<code class="doxybook">
<span>static const detail::host_t <b>host</b>;</span></code>
<code>thrust::host</code> is the default parallel execution policy associated with Thrust's host backend system configured by the <code>THRUST&#95;HOST&#95;SYSTEM</code> macro.

Instead of relying on implicit algorithm dispatch through iterator system tags, users may directly target algorithm dispatch at Thrust's host system by providing <code>thrust::host</code> as an algorithm parameter.

Explicit dispatch can be useful in avoiding the introduction of data copies into containers such as <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1host__vector.html">thrust::host&#95;vector</a></code>.

Note that even though <code>thrust::host</code> targets the host CPU, it is a parallel execution policy. That is, the order that an algorithm invokes functors or dereferences iterators is not defined.

The type of <code>thrust::host</code> is implementation-defined.

The following code snippet demonstrates how to use <code>thrust::host</code> to explicitly dispatch an invocation of <code>thrust::for&#95;each</code> to the host backend system:



```cpp
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <cstdio>

struct printf_functor
{
  __host__ __device__
  void operator()(int x)
  {
    printf("%d\n", x);
  }
};
...
int vec(3);
vec[0] = 0; vec[1] = 1; vec[2] = 2;

thrust::for_each(thrust::host, vec.begin(), vec.end(), printf_functor());

// 0 1 2 is printed to standard output in some unspecified order
```

**See**:
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1host__execution__policy.html">host_execution_policy</a>
* thrust::device 

<h3 id="variable-device">
Variable <code>thrust::device</code>
</h3>

<code class="doxybook">
<span>THRUST_INLINE_CONSTANT detail::device_t <b>device</b>;</span></code>
<code>thrust::device</code> is the default parallel execution policy associated with Thrust's device backend system configured by the <code>THRUST&#95;DEVICE&#95;SYSTEM</code> macro.

Instead of relying on implicit algorithm dispatch through iterator system tags, users may directly target algorithm dispatch at Thrust's device system by providing <code>thrust::device</code> as an algorithm parameter.

Explicit dispatch can be useful in avoiding the introduction of data copies into containers such as <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__vector.html">thrust::device&#95;vector</a></code> or to avoid wrapping e.g. raw pointers allocated by the CUDA API with types such as <code><a href="{{ site.baseurl }}/api/classes/classthrust_1_1device__ptr.html">thrust::device&#95;ptr</a></code>.

The user must take care to guarantee that the iterators provided to an algorithm are compatible with the device backend system. For example, raw pointers allocated by <code>std::malloc</code> typically cannot be dereferenced by a GPU. For this reason, raw pointers allocated by host APIs should not be mixed with a <code>thrust::device</code> algorithm invocation when the device backend is CUDA.

The type of <code>thrust::device</code> is implementation-defined.

The following code snippet demonstrates how to use <code>thrust::device</code> to explicitly dispatch an invocation of <code>thrust::for&#95;each</code> to the device backend system:



```cpp
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <cstdio>

struct printf_functor
{
  __host__ __device__
  void operator()(int x)
  {
    printf("%d\n", x);
  }
};
...
thrust::device_vector<int> vec(3);
vec[0] = 0; vec[1] = 1; vec[2] = 2;

thrust::for_each(thrust::device, vec.begin(), vec.end(), printf_functor());

// 0 1 2 is printed to standard output in some unspecified order
```

**See**:
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1host__execution__policy.html">host_execution_policy</a>
* thrust::device 


