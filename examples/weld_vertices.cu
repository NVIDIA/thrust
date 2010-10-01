#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>
#include <thrust/sort.h>

#include <iostream>

/*
 * This example "welds" triangle vertices together by taking as
 * input "triangle soup" and eliminating redundant vertex positions
 * and shared edges.  A connected mesh is the result.
 * 
 *
 * Input: 9 vertices representing a mesh with 3 triangles
 *  
 *  Mesh              Vertices 
 *    ------           (2)      (5)--(4)    (8)      
 *    | \ 2| \          | \       \   |      | \
 *    |  \ |  \   <->   |  \       \  |      |  \
 *    | 0 \| 1 \        |   \       \ |      |   \
 *    -----------      (0)--(1)      (3)    (6)--(7)
 *
 *   (vertex 1 equals vertex 3, vertex 2 equals vertex 5, ...)
 *
 * Output: mesh representation with 5 vertices and 9 indices
 *
 *  Vertices            Indices
 *   (1)--(3)            [(0,2,1),
 *    | \  | \            (2,3,1), 
 *    |  \ |  \           (2,4,3)]
 *    |   \|   \
 *   (0)--(2)--(4)
 */

// compare two float2s for equality
struct float2_equal_to
{
    __host__ __device__
    bool operator()(float2 a, float2 b)
    {
        return a.x == b.x && a.y == b.y;
    }
};

// compare ordering of two float2s
struct float2_less
{
    __host__ __device__
    bool operator()(float2 a, float2 b)
    {
        if (a.x < b.x)
            return true;
        else if (a.x > b.x)
            return false;

        return a.y < b.y;
    }
};

int main(void)
{
    // allocate memory for input mesh representation
    thrust::device_vector<float2> input(9);

    input[0] = make_float2(0,0);  // First Triangle
    input[1] = make_float2(1,0);
    input[2] = make_float2(0,1);
    input[3] = make_float2(1,0);  // Second Triangle
    input[4] = make_float2(1,1);
    input[5] = make_float2(0,1);
    input[6] = make_float2(1,0);  // Third Triangle
    input[7] = make_float2(2,0);
    input[8] = make_float2(1,1);

    // allocate space for output mesh representation
    thrust::device_vector<float2>       vertices = input;
    thrust::device_vector<unsigned int> indices(input.size());

    // sort vertices to bring duplicates together
    thrust::sort(vertices.begin(), vertices.end(), float2_less());

    // find unique vertices and erase redundancies
    vertices.erase(thrust::unique(vertices.begin(), vertices.end(), float2_equal_to()), vertices.end());

    // find index of each input vertex in the list of unique vertices
    thrust::lower_bound(vertices.begin(), vertices.end(),
                        input.begin(), input.end(),
                        indices.begin(),
                        float2_less());

    // print output mesh representation
    std::cout << "Output Representation" << std::endl;
    for(size_t i = 0; i < vertices.size(); i++)
    {
        float2 v = vertices[i];
        std::cout << " vertices[" << i << "] = (" << v.x << "," << v.y << ")" << std::endl;
    }
    for(size_t i = 0; i < indices.size(); i++)
    {
        std::cout << " indices[" << i << "] = " << indices[i] << std::endl;
    }

    return 0;
}

