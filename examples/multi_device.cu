#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/iterator/detail/placement/place.h>
#include <thrust/iterator/detail/placement/make_unplaced_iterator.h>
#include <cassert>
#include <thrust/device_malloc_allocator.h>
#include <thrust/detail/placed_allocator.h>

typedef thrust::device_malloc_allocator<int> base_allocator_type;
typedef thrust::detail::placed_allocator<int,base_allocator_type> placed_allocator;
typedef thrust::device_vector<int,placed_allocator> device_vector_with_place;

int reduce(const thrust::host_vector<device_vector_with_place> &vectors)
{
  int result = 0;

  for(int i = 0; i < vectors.size(); ++i)
  {
    thrust::detail::push_place(vectors[i].get_allocator().get_place());
    result += thrust::reduce(thrust::detail::make_unplaced_iterator(vectors[i].begin()),
                             thrust::detail::make_unplaced_iterator(vectors[i].end()));
    thrust::detail::pop_place();
  }

  return result;
}

int main(void)
{
  int N = 1000;

  int num_places = thrust::detail::num_places();

  thrust::host_vector<device_vector_with_place> vectors(num_places);

  for(int i = 0; i < num_places; ++i)
  {
    vectors[i].resize(N);

    thrust::detail::push_place(vectors[i].get_allocator().get_place());
    thrust::fill(vectors[i].begin(), vectors[i].end(), 13);
    thrust::detail::pop_place();
  }

  int result = reduce(vectors);
  assert(result == 13 * N * num_places);

  std::cout << "result: " << result << std::endl;
  return 0;
}

