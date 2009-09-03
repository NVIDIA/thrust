#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>

#include <cuda.h>   // for float2
#include <stdlib.h> // for rand()
#include <iostream>

// return a random float2 in [0,1)^2
float2 make_random_float2(void)
{
  return make_float2(static_cast<float>(min(RAND_MAX-1,rand())) / (RAND_MAX),
                     static_cast<float>(min(RAND_MAX-1,rand())) / (RAND_MAX));
}

// hash a point in the unit square to the index of
// the grid bucket that contains it
struct point_to_bucket_index
{
  __host__ __device__
  point_to_bucket_index(unsigned int width, unsigned int height)
    :w(width),h(height){}

  __host__ __device__
  unsigned int operator()(float2 p) const
  {
    // find the raster indices of p's bucket
    unsigned int x = p.x * w;
    unsigned int y = p.y * h;

    // return the bucket's linear index
    return y * w + x;
  }

  unsigned int w, h;
};

int main(void)
{
  const size_t N = 100000;

  // allocate some random points in the unit square on the host
  thrust::host_vector<float2> h_points(N);
  thrust::generate(h_points.begin(), h_points.end(), make_random_float2);

  // transfer to device
  thrust::device_vector<float2> points = h_points;

  // allocate storage for a 2D grid
  // of dimensions w x h
  unsigned int w = 200, h = 100;

  // the grid data structure keeps a range per grid bucket:
  // each bucket_begin[i] indexes the first element of bucket i's list of points
  // each bucket_end[i] indexes one past the last element of bucket i's list of points
  thrust::device_vector<unsigned int> bucket_begin(w*h);
  thrust::device_vector<unsigned int> bucket_end(w*h);

  // allocate storage for each point's bucket index
  thrust::device_vector<unsigned int> bucket_indices(N);

  // transform the points to their bucket indices
  thrust::transform(points.begin(),
                    points.end(),
                    bucket_indices.begin(),
                    point_to_bucket_index(w,h));

  // sort the points by their bucket index
  thrust::sort_by_key(bucket_indices.begin(),
                      bucket_indices.end(),
                      points.begin());

  // find the beginning of each bucket's list of points
  thrust::counting_iterator<unsigned int> search_begin(0);
  thrust::lower_bound(bucket_indices.begin(),
                      bucket_indices.end(),
                      search_begin,
                      search_begin + w*h,
                      bucket_begin.begin());

  // find the end of each bucket's list of points
  thrust::upper_bound(bucket_indices.begin(),
                      bucket_indices.end(),
                      search_begin,
                      search_begin + w*h,
                      bucket_end.begin());

  // write out bucket (150, 50)'s list of points
  unsigned int bucket_idx = 50 * w + 150;
  std::cout << "bucket (150, 50)'s list of points:" << std::endl;
  for(unsigned int point_idx = bucket_begin[bucket_idx];
      point_idx != bucket_end[bucket_idx];
      ++point_idx)
  {
    float2 p = points[point_idx];
    std::cout << "(" << p.x << "," << p.y << ")" << std::endl;
  }

  return 0;
}

