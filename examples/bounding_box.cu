#include <thrust/transform_reduce.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>

#include <cuda.h>    // for float2
#include <stdlib.h>  // for rand()


// This example shows how to compute a bounding box
// for a set of points in two dimensions.


// bounding box type
typedef thrust::pair<float2, float2> bbox;

// reduce a pair of bounding boxes (a,b) to a bounding box containing a and b
struct bbox_reduction
{
    __host__ __device__
    bbox operator()(bbox a, bbox b)
    {
        // lower left corner
        float2 ll = make_float2(min(a.first.x, b.first.x), min(a.first.y, b.first.y));
        
        // upper right corner
        float2 ur = make_float2(max(a.second.x, b.second.x), max(a.second.y, b.second.y));

        return bbox(ll, ur);
    }
};

// convert a point to a bbox containing that point, (point) -> (point, point)
struct bbox_transformation
{
    __host__ __device__
    bbox operator()(float2 point)
    {
        return bbox(point, point);
    }
};

int main(void)
{
    const size_t N = 40;
    
    // allocate storage for points
    thrust::device_vector<float2> points(N);

    // generate some random points in the unit square
    for(size_t i = 0; i < N; i++)
        points[i] = make_float2( ((float) rand() / (RAND_MAX + 1.0)), ((float) rand() / (RAND_MAX + 1.0)) );

    // initial bounding box contains first point
    bbox init = bbox(points[0], points[0]);
    
    // transformation operation
    bbox_transformation unary_op;

    // binary reduction operation
    bbox_reduction binary_op;
    
    // compute the bounding box for the point set
    bbox result = thrust::transform_reduce(points.begin(), points.end(), unary_op, init, binary_op);

    // print output
    std::cout << "bounding box ";
    std::cout << "(" << result.first.x  << "," << result.first.y  << ") ";
    std::cout << "(" << result.second.x << "," << result.second.y << ")" << std::endl;

    return 0;
}
