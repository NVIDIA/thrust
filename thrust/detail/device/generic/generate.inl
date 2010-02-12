/*
 *  Copyright 2008-2010 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#include <thrust/detail/device/for_each.h>

namespace thrust
{
namespace detail
{
namespace device
{
namespace generic
{
namespace detail
{

template<typename Generator>
struct generate_functor
{
    Generator gen;

    generate_functor(Generator _gen)
        : gen(_gen){}

    template<typename T>
        __host__ __device__
        void operator()(T &x)
        {
            x = gen();
        }
}; // end generate_functor
  
} // end namespace detail


template<typename ForwardIterator,
         typename Generator>
  void generate(ForwardIterator first,
                ForwardIterator last,
                Generator gen)
{
    detail::generate_functor<Generator> f(gen);
    thrust::detail::device::for_each(first, last, f);
} // end generate()


} // end namespace generic
} // end namespace device
} // end namespace detail
} // end namespace thrust

