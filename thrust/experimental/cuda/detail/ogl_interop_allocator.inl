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

#include <thrust/experimental/cuda/ogl_interop_allocator.h>

// define this static data here to WAR linkage issues with older gcc
template<typename T> std::map<typename ogl_interop_allocator<T>::pointer, GLuint> ogl_interop_allocator<T>::s_pointer_to_buffer_object;

template<typename T>
  ogl_interop_allocator<T>
    ::ogl_interop_allocator(void)
{
  ;
} // end ogl_interop_allocator::ogl_interop_allocator()

template<typename T>
  ogl_interop_allocator<T>
    ::ogl_interop_allocator(const ogl_interop_allocator &a)
{
  ;
} // end ogl_interop_allocator::ogl_interop_allocator()

template<typename T>
  template<typename U>
    ogl_interop_allocator<T>
      ::ogl_interop_allocator(const ogl_interop_allocator<U> &a)
{
  ;
} // end ogl_interop_allocator::ogl_interop_allocator()

template<typename T>
  typename ogl_interop_allocator<T>::pointer
    ogl_interop_allocator<T>
      ::address(reference r)
{
  return &r;
} // end ogl_interop_allocator::address()

template<typename T>
  typename ogl_interop_allocator<T>::const_pointer
    ogl_interop_allocator<T>
      ::address(const_reference r)
{
  return &r;
} // end ogl_interop_allocator::address()

template<typename T>
  typename ogl_interop_allocator<T>::pointer
    ogl_interop_allocator<T>
      ::allocate(size_type cnt, const_pointer)
{
  // create a GL buffer object
  GLuint buffer = 0;
  glGenBuffers(1, &buffer);

  // XXX check GL error
  if(glGetError())
  {
    std::cerr << "ogl_interop_allocator::allocate(): GL error after glGenBuffers" << std::endl;
  }

  // bind the buffer object
  glBindBuffer(GL_ARRAY_BUFFER, buffer);
  // XXX check GL error
  if(glGetError())
  {
    std::cerr << "ogl_interop_allocator::allocate(): GL error after glBindBuffer" << std::endl;
  }

  // XXX need to push/pop the correct GL state
  glBufferData(GL_ARRAY_BUFFER, cnt * sizeof(value_type), 0, GL_DYNAMIC_DRAW);

  // XXX check GL error
  if(glGetError())
  {
    std::cerr << "ogl_interop_allocator::allocate(): GL error after glBufferData" << std::endl;
  }

  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // XXX check GL error
  if(glGetError())
  {
    std::cerr << "ogl_interop_allocator::allocate(): GL error after glBindBuffer" << std::endl;
  }
  
  // register buffer object with CUDA
  cudaError_t cuda_error = cudaGLRegisterBufferObject(buffer);

  // XXX check CUDA error
  if(cuda_error)
  {
    std::cerr << "ogl_interop_allocator::allocate(): CUDA error after cudaGLRegisterBufferObject" << std::endl;
  }

  // get the pointer from CUDA
  T *raw_ptr = 0;
  cuda_error = cudaGLMapBufferObject((void**)&raw_ptr, buffer);

  // XXX check CUDA error
  if(cuda_error)
  {
    std::cerr << "ogl_interop_allocator::allocate(): CUDA error after cudaGLMapBufferObject" << std::endl;
  }

  // make a pointer
  pointer result = thrust::device_pointer_cast(raw_ptr);

  // remember the mapping from ptr to buffer
  s_pointer_to_buffer_object.insert(std::make_pair(result, buffer));

  return result;
} // end ogl_interop_allocator::allocate()

template<typename T>
  void ogl_interop_allocator<T>
    ::deallocate(pointer p, size_type cnt)
{
  // look up the buffer id
  typename std::map<pointer, GLuint>::iterator buffer
    = s_pointer_to_buffer_object.find(p);

  // XXX throw an exception or something if p doesn't exist
  // if(buffer == s_pointer_to_buffer_object.end()) ...

  // unmap the buffer object
  cudaGLUnmapBufferObject(buffer->second);

  // XXX check CUDA error

  // delete the buffer object
  glDeleteBuffers(1, &buffer->second);

  // XXX check GL error

  // erase the entry
  s_pointer_to_buffer_object.erase(buffer);
} // end ogl_interop_allocator::deallocate()


template<typename T>
  typename ogl_interop_allocator<T>::size_type
    ogl_interop_allocator<T>
      ::max_size(void) const
{
  // XXX query ogl for the maximum size buffer
  return 1000000000;
} // end ogl_interop_allocator::max_size()


template<typename T>
  bool ogl_interop_allocator<T>
    ::operator==(const ogl_interop_allocator &rhs)
{
  return true;
} // end ogl_interop_allocator::operator==()


template<typename T>
  bool ogl_interop_allocator<T>
    ::operator!=(const ogl_interop_allocator &rhs)
{
  return !(*this == rhs);
} // end ogl_interop_allocator::operator!=()


template<typename T>
  GLuint ogl_interop_allocator<T>
    ::get_buffer_object(pointer ptr)
{
  // look up the buffer id
  typename std::map<pointer, GLuint>::iterator result
    = s_pointer_to_buffer_object.find(ptr);

  if(result == s_pointer_to_buffer_object.end())
  {
    return 0;
  } // end if

  return result->second;
} // end ogl_interop_allocator::get_buffer_object()


