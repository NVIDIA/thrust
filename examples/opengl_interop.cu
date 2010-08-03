#include <math.h>

#if defined(_MSC_VER)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#if defined(__APPLE__)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif // __APPLE__

#include <cuda_gl_interop.h>

#include <thrust/device_vector.h>
#include <thrust/experimental/cuda/ogl_interop_allocator.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

// This example, adapted from the CUDA SDK sample program "simpleGL", demonstrates
// how to use the experimental ogl_interop_allocator with device_vector to allow
// device_vector to manage OpenGL memory resources. The functionality of
// ogl_interop_allocator relies on behavior currently unsupported by the NVIDIA
// driver and should not be relied upon in production code.

// constants
const unsigned int g_window_width = 512;
const unsigned int g_window_height = 512;

const unsigned int g_mesh_width = 256;
const unsigned int g_mesh_height = 256;

// for a device_vector interoperable with OpenGL, pass ogl_interop_allocator as the allocator type
typedef thrust::device_vector<float4, thrust::experimental::cuda::ogl_interop_allocator<float4> > gl_vector;

// define these variables in the global scope so that the GLUT callback functions have access to it
gl_vector g_vec;

float g_anim = 0.0;

// mouse controls
int g_mouse_old_x, g_mouse_old_y;
int g_mouse_buttons = 0;
float g_rotate_x = 0.0, g_rotate_y = 0.0;
float g_translate_z = -3.0;

struct sine_wave
{
  sine_wave(unsigned int w, unsigned int h, float t)
    : width(w), height(h), time(t) {}

  __host__ __device__
  float4 operator()(unsigned int i)
  {
    unsigned int x = i % width;
    unsigned int y = i / width;

    // calculate uv coordinates
    float u = x / (float) width;
    float v = y / (float) height;
    u = u*2.0f - 1.0f;
    v = v*2.0f - 1.0f;

    // calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

    // write output vertex
    return make_float4(u, w, v, 1.0f);
  } // end operator()

  float time;
  unsigned int width, height;
}; // end sine_wave

bool init_gl(void)
{
  // default initialization
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glDisable(GL_DEPTH_TEST);

  // viewport
  glViewport(0, 0, g_window_width, g_window_height);

  // projection
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, (GLfloat)g_window_width / (GLfloat) g_window_height, 0.1, 10.0);

  return true;
} // end init_gl

void display(void)
{
  // transform the mesh
  thrust::counting_iterator<int,thrust::device_space_tag> first(0);
  thrust::counting_iterator<int,thrust::device_space_tag> last(g_mesh_width * g_mesh_height);

  thrust::transform(first, last,
                    g_vec.begin(),
                    sine_wave(g_mesh_width,g_mesh_height,g_anim));

  // map the vector into GL
  thrust::device_ptr<float4> ptr = &g_vec[0];

  // pass the device_ptr to the allocator's static function map_buffer
  // to map it into GL
  GLuint buffer = gl_vector::allocator_type::map_buffer(ptr);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // set view matrix
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(0.0, 0.0, g_translate_z);
  glRotatef(g_rotate_x, 1.0, 0.0, 0.0);
  glRotatef(g_rotate_y, 0.0, 1.0, 0.0);

  // render from the vbo
  glBindBuffer(GL_ARRAY_BUFFER, buffer);
  glVertexPointer(4, GL_FLOAT, 0, 0);

  glEnableClientState(GL_VERTEX_ARRAY);
  glColor3f(1.0, 0.0, 0.0);
  glDrawArrays(GL_POINTS, 0, g_mesh_width * g_mesh_height);
  glDisableClientState(GL_VERTEX_ARRAY);

  glutSwapBuffers();
  glutPostRedisplay();

  g_anim += 0.001;

  // unmap the vector from GL
  gl_vector::allocator_type::unmap_buffer(buffer);
} // end display

void mouse(int button, int state, int x, int y)
{
  if(state == GLUT_DOWN)
  {
    g_mouse_buttons |= 1<<button;
  } // end if
  else if(state == GLUT_UP)
  {
    g_mouse_buttons = 0;
  } // end else if

  g_mouse_old_x = x;
  g_mouse_old_y = y;
  glutPostRedisplay();
} // end mouse

void motion(int x, int y)
{
  float dx, dy;
  dx = x - g_mouse_old_x;
  dy = y - g_mouse_old_y;

  if(g_mouse_buttons & 1)
  {
    g_rotate_x += dy * 0.2;
    g_rotate_y += dx * 0.2;
  } // end if
  else if(g_mouse_buttons & 4)
  {
    g_translate_z += dy * 0.01;
  } // end else if

  g_mouse_old_x = x;
  g_mouse_old_y = y;
} // end motion

void keyboard(unsigned char key, int, int)
{
  switch(key)
  {
    // catch 'esc'
    case(27):
      // deallocate memory
      g_vec.clear();
      g_vec.shrink_to_fit();
      exit(0);
    default:
      break;
  } // end switch
} // end keyboard

int main(int argc, char** argv)
{
  // Create GL context
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(g_window_width, g_window_height);
  glutCreateWindow("Thrust/GL interop");

  // initialize GL
  if(!init_gl())
  {
    throw std::runtime_error("Couldn't initialize OpenGL");
  } // end if

  // register callbacks
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);

  // resize the vector to fit the mesh
  g_vec.resize(g_mesh_width * g_mesh_height);

  // transform the mesh
  thrust::counting_iterator<int,thrust::device_space_tag> first(0);
  thrust::counting_iterator<int,thrust::device_space_tag> last(g_mesh_width * g_mesh_height);

  thrust::transform(first, last,
                    g_vec.begin(),
                    sine_wave(g_mesh_width,g_mesh_height,g_anim));

  // start rendering mainloop
  glutMainLoop();

  return 0;
} // end main

