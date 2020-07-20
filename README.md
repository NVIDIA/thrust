Thrust: Code at the speed of light
==================================

Thrust is a C++ parallel programming library which resembles the C++ Standard
Library. Thrust's **high-level** interface greatly enhances
programmer **productivity** while enabling performance portability between
GPUs and multicore CPUs. **Interoperability** with established technologies
(such as CUDA, TBB, and OpenMP) facilitates integration with existing
software. Develop **high-performance** applications rapidly with Thrust!

Thrust is included in the NVIDIA HPC SDK and the CUDA Toolkit.

Refer to the [Quick Start Guide](http://github.com/thrust/thrust/wiki/Quick-Start-Guide) page for further information and examples.

Examples
--------

Thrust is best explained through examples. The following source code
generates random numbers serially and then transfers them to a parallel
device where they are sorted.

```c++
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

int main(void)
{
  // generate 32M random numbers serially
  thrust::host_vector<int> h_vec(32 << 20);
  std::generate(h_vec.begin(), h_vec.end(), rand);

  // transfer data to the device
  thrust::device_vector<int> d_vec = h_vec;

  // sort data on the device (846M keys per second on GeForce GTX 480)
  thrust::sort(d_vec.begin(), d_vec.end());

  // transfer data back to host
  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

  return 0;
}
```

This code sample computes the sum of 100 random numbers in parallel:

```c++
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <algorithm>
#include <cstdlib>

int main(void)
{
  // generate random data serially
  thrust::host_vector<int> h_vec(100);
  std::generate(h_vec.begin(), h_vec.end(), rand);

  // transfer to device and compute sum
  thrust::device_vector<int> d_vec = h_vec;
  int x = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());
  return 0;
}
```

Releases
--------

Thrust is distributed with the NVIDIA HPC SDK and the CUDA Toolkit in addition
to GitHub.

See the [changelog](CHANGELOG.md) for details about specific releases.

| Thrust Release    | Included In                             |
| ----------------- | --------------------------------------- |
| 1.9.10-1          | NVIDIA HPC SDK 20.7 & CUDA Toolkit 11.1 |
| 1.9.10            | NVIDIA HPC SDK 20.5                     |
| 1.9.9             | CUDA Toolkit 11.0                       |
| 1.9.8-1           | NVIDIA HPC SDK 20.3                     |
| 1.9.8             | CUDA Toolkit 11.0 Early Access          |
| 1.9.7-1           | CUDA Toolkit 10.2 for Tegra             |
| 1.9.7             | CUDA Toolkit 10.2                       |
| 1.9.6-1           | NVIDIA HPC SDK 20.3                     |
| 1.9.6             | CUDA Toolkit 10.1 Update 2              |
| 1.9.5             | CUDA Toolkit 10.1 Update 1              |
| 1.9.4             | CUDA Toolkit 10.1                       |
| 1.9.3             | CUDA Toolkit 10.0                       |
| 1.9.2             | CUDA Toolkit 9.2                        |
| 1.9.1-2           | CUDA Toolkit 9.1                        |
| 1.9.0-5           | CUDA Toolkit 9.0                        |
| 1.8.3             | CUDA Toolkit 8.0                        |
| 1.8.2             | CUDA Toolkit 7.5                        |
| 1.8.1             | CUDA Toolkit 7.0                        |
| 1.8.0             |                                         |
| 1.7.2             | CUDA Toolkit 6.5                        |
| 1.7.1             | CUDA Toolkit 6.0                        |
| 1.7.0             | CUDA Toolkit 5.5                        |
| 1.6.0             |                                         |
| 1.5.3             | CUDA Toolkit 5.0                        |
| 1.5.2             | CUDA Toolkit 4.2                        |
| 1.5.1             | CUDA Toolkit 4.1                        |
| 1.5.0             |                                         |
| 1.4.0             | CUDA Toolkit 4.0                        |
| 1.3.0             |                                         |
| 1.2.1             |                                         |
| 1.2.0             |                                         |
| 1.1.1             |                                         |
| 1.1.0             |                                         |
| 1.0.0             |                                         |

Adding Thrust To A CMake Project
--------------------------------

Since Thrust is a header library, there is no need to build or install Thrust
to use it. The `thrust` directory contains a complete, ready-to-use Thrust
package upon checkout.

We provide CMake configuration files that make it easy to include Thrust
from other CMake projects. See the [CMake README](thrust/cmake/README.md)
for details.

Development Process
-------------------

Thrust uses the [CMake build system](https://cmake.org/) to build unit tests,
examples, and header tests. To build Thrust as a developer, the following
recipe should be followed:

```
# Clone Thrust and CUB repos recursively:
git clone --recursive https://github.com/thrust/thrust.git
cd thrust

# Create build directory:
mkdir build
cd build

# Configure -- use one of the following:
cmake ..   # Command line interface.
ccmake ..  # ncurses GUI (Linux only)
cmake-gui  # Graphical UI, set source/build directories in the app

# Build:
cmake --build . -j <num jobs>   # invokes make (or ninja, etc)

# Run tests and examples:
ctest
```

By default, a serial `CPP` host system, `CUDA` accelerated device system, and
C++14 standard are used. This can be changed in CMake. More information on
configuring your Thrust build and creating a pull request can be found in
[CONTRIBUTING.md](CONTRIBUTING.md).
