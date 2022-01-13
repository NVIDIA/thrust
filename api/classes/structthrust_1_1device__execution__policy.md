---
title: thrust::device_execution_policy
parent: Parallel Execution Policies
nav_exclude: true
has_children: true
has_toc: false
---

# Struct `thrust::device_execution_policy`

<code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1device__execution__policy.html">device&#95;execution&#95;policy</a></code> is the base class for all Thrust parallel execution policies which are derived from Thrust's default device backend system configured with the <code>THRUST&#95;DEVICE&#95;SYSTEM</code> macro.

Custom user-defined backends which wish to inherit the functionality of Thrust's device backend system should derive a policy from this type in order to interoperate with Thrust algorithm dispatch.

The following code snippet demonstrates how to derive a standalone custom execution policy from <code><a href="{{ site.baseurl }}/api/classes/structthrust_1_1device__execution__policy.html">thrust::device&#95;execution&#95;policy</a></code> to implement a backend which specializes <code>for&#95;each</code> while inheriting the behavior of every other algorithm from the device system:



```cpp
#include <thrust/execution_policy.h>
#include <iostream>

// define a type derived from thrust::device_execution_policy to distinguish our custom execution policy:
struct my_policy : thrust::device_execution_policy<my_policy> {};

// overload for_each on my_policy
template<typename Iterator, typename Function>
Iterator for_each(my_policy, Iterator first, Iterator last, Function f)
{
  std::cout << "Hello, world from for_each(my_policy)!" << std::endl;

  for(; first < last; ++first)
  {
    f(*first);
  }

  return first;
}

struct ignore_argument
{
  void operator()(int) {}
};

int main()
{
  int data[4];

  // dispatch thrust::for_each using our custom policy:
  my_policy exec;
  thrust::for_each(exec, data, data + 4, ignore_argument());

  // dispatch thrust::transform whose behavior our policy inherits
  thrust::transform(exec, data, data, + 4, data, thrust::identity<int>());

  return 0;
}
```

**Inherits From**:
`thrust::system::__THRUST_DEVICE_SYSTEM_NAMESPACE::execution_policy< DerivedPolicy >`

**See**:
* execution_policy 
* <a href="{{ site.baseurl }}/api/classes/structthrust_1_1host__execution__policy.html">host_execution_policy</a>

<code class="doxybook">
<span>#include <thrust/execution_policy.h></span><br>
<span>template &lt;typename DerivedPolicy&gt;</span>
<span>struct thrust::device&#95;execution&#95;policy {</span>
<span>};</span>
</code>

