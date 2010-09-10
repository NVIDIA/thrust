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
#include <cuda_gl_interop.h>
#include <thrust/system/cuda_error.h>
#include <limits>

// declare these functions here to avoid cross-platform GL header badness
// #including cuda_gl_interop.h above ensures things like GLAPI, APIENTRY, GLenum, etc are defined
// XXX this probably isn't even portable
extern "C"
{

GLAPI void APIENTRY glBindBuffer(GLenum, GLuint);
GLAPI void APIENTRY glDeleteBuffers(GLsizei, const GLuint *);
GLAPI void APIENTRY glGenBuffers(GLsizei, GLuint *);
GLAPI void APIENTRY glBufferData(GLenum, GLsizeiptr, const GLvoid *, GLenum);

} // end extern "C"

namespace thrust
{

namespace experimental
{

namespace cuda
{

// define this static data here to WAR linkage issues with older gcc
template<typename T> std::map<typename ogl_interop_allocator<T>::pointer, GLuint> ogl_interop_allocator<T>::s_pointer_to_buffer_object;
template<typename T> std::map<GLuint, typename ogl_interop_allocator<T>::pointer> ogl_interop_allocator<T>::s_buffer_object_to_pointer;

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

  // check GL error
  if(GLenum gl_error = glGetError())
  {
    throw std::runtime_error("ogl_interop_allocator::allocate(): error after glGenBuffers");
  } // end if

  // bind the buffer object
  glBindBuffer(GL_ARRAY_BUFFER, buffer);

  // check GL error
  if(GLenum gl_error = glGetError())
  {
    throw std::runtime_error("ogl_interop_allocator::allocate(): error after glBindBuffer");
  } // end if

  // XXX need to push/pop the correct GL state
  glBufferData(GL_ARRAY_BUFFER, cnt * sizeof(value_type), 0, GL_DYNAMIC_DRAW);

  // check GL error
  if(GLenum gl_error = glGetError())
  {
    throw std::runtime_error("ogl_interop_allocator::allocate(): error after glBufferData");
  } // end if

  // XXX instead of leaving GL in this weird state, push/pop it above

  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // check GL error
  if(glGetError())
  {
    throw std::runtime_error("ogl_interop_allocator::allocate(): error after glBindBuffer");
  } // end if
  
  // register buffer object with CUDA
  cudaError_t cuda_error = cudaGLRegisterBufferObject(buffer);

  // check CUDA error
  if(cuda_error)
  {
    throw thrust::system_error(cuda_error, thrust::cuda_category(), "ogl_interop_allocator::allocate(): error after cudaGLRegisterBufferObject");
  } // end if

  // get the pointer from CUDA
  T *raw_ptr = 0;
  cuda_error = cudaGLMapBufferObject((void**)&raw_ptr, buffer);

  // check CUDA error
  if(cuda_error)
  {
    throw thrust::system_error(cuda_error, thrust::cuda_category(), "ogl_interop_allocator::allocate(): error after cudaGLMapBufferObject");
  } // end if

  // make a pointer
  pointer result = thrust::device_pointer_cast(raw_ptr);

  // remember the mapping from ptr to buffer
  s_pointer_to_buffer_object.insert(std::make_pair(result, buffer));

  // remember the mapping from buffer to ptr
  s_buffer_object_to_pointer.insert(std::make_pair(buffer, result));

  return result;
} // end ogl_interop_allocator::allocate()

template<typename T>
  void ogl_interop_allocator<T>
    ::deallocate(pointer p, size_type cnt)
{
  // look up the buffer id
  typename std::map<pointer, GLuint>::iterator ptr_and_buffer
    = s_pointer_to_buffer_object.find(p);

  // throw an exception if p is invalid
  if(ptr_and_buffer == s_pointer_to_buffer_object.end())
  {
    throw std::runtime_error("ogl_interop_allocator::deallocate(): invalid pointer");
  } // end if

  typename std::map<GLuint, pointer>::iterator buffer_and_ptr
    = s_buffer_object_to_pointer.find(ptr_and_buffer->second);

  // throw an exception if we can't find the inverse mapping
  if(buffer_and_ptr == s_buffer_object_to_pointer.end())
  {
    throw std::runtime_error("ogl_interop_allocator::deallocate(): mapping is in inconsistent state");
  } // end if

  // unmap the buffer object
  cudaError_t cuda_error = cudaGLUnmapBufferObject(ptr_and_buffer->second);

  // check CUDA error
  if(cuda_error)
  {
    throw thrust::system_error(cuda_error, thrust::cuda_category(), "ogl_interop_allocator::deallocate(): error after cudaGLUnmapBufferObject");
  } // end if

  // delete the buffer object
  glDeleteBuffers(1, &ptr_and_buffer->second);

  // check GL error
  if(GLenum gl_error = glGetError())
  {
    throw std::runtime_error("ogl_interop_allocator::deallocate(): error after glDeleteBuffers");
  } // end if

  // erase the entries
  s_pointer_to_buffer_object.erase(ptr_and_buffer);
  s_buffer_object_to_pointer.erase(buffer_and_ptr);
} // end ogl_interop_allocator::deallocate()


template<typename T>
  typename ogl_interop_allocator<T>::size_type
    ogl_interop_allocator<T>
      ::max_size(void) const
{
  // XXX query ogl for the maximum size buffer
  return std::numeric_limits<size_type>::max THRUST_PREVENT_MACRO_SUBSTITUTION ();
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
    ::map_buffer(pointer ptr)
{
  // look up the buffer id
  typename std::map<pointer, GLuint>::iterator result
    = s_pointer_to_buffer_object.find(ptr);

  if(result == s_pointer_to_buffer_object.end())
  {
    throw std::runtime_error("ogl_interop_allocator::map_buffer(): invalid ptr");
  } // end if

  // unmap the buffer from cuda
  cudaError_t cuda_error = cudaGLUnmapBufferObject(result->second);

  // check CUDA error
  if(cuda_error)
  {
    throw thrust::system_error(cuda_error, thrust::cuda_category(), "ogl_interop_allocator::map_buffer(): error after cudaGLUnmapBufferObject");
  } // end if

  return result->second;
} // end ogl_interop_allocator::map_buffer()


template<typename T>
  void ogl_interop_allocator<T>
    ::unmap_buffer(GLuint buffer)
{
  // find the old pointer associated with buffer
  typename std::map<GLuint, pointer>::iterator old_mapping
    = s_buffer_object_to_pointer.find(buffer);

  // throw an exception if the buffer wasn't found in the map
  if(old_mapping == s_buffer_object_to_pointer.end())
  {
    throw std::runtime_error("ogl_interop_allocator::unmap_buffer(): invalid buffer");
  } // end if

  // remap the buffer into cuda
  value_type *new_ptr = 0;
  cudaError_t error = cudaGLMapBufferObject((void**)&new_ptr, buffer);
  
  // check CUDA error
  if(error)
  {
    throw thrust::system_error(error, thrust::cuda_category());
  } // end if

  // if a new mapping occurred, throw an exception
  // the assumption is that the buffer is "pinned" to a CUDA address that shall not change
  if(thrust::device_pointer_cast(new_ptr) != old_mapping->second)
  {
    throw std::runtime_error("ogl_interop_allocator::unmap_buffer(): operation resulted in a new mapping");
  } // end if
} // end ogl_interop_allocator::unmap_buffer()

} // end cuda

} // end experimental

} // end thrust

