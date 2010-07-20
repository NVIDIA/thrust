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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/iterator/detail/backend_iterator_spaces.h>

namespace thrust
{

namespace detail
{

namespace place_detail
{

template<typename Space>
  class place
{
  public:
    inline place(int resource = 0)
      : m_resource(resource) {}

    typedef Space space;

    __host__ __device__
    bool operator==(const place &rhs) const
    {
      return m_resource == rhs.m_resource;
    }

  private:
    int m_resource;

    friend struct place_core_access;
};

} // end place_detail

typedef place_detail::place<thrust::detail::default_device_space_tag> place;

inline void push_place(place p);

inline void pop_place(void);

inline place get_current_place(void);

inline size_t num_places(void);

} // end detail

} // end thrust

#include <thrust/iterator/detail/placement/place.inl>

