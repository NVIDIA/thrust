#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/functional.h>

// for printing
#include <thrust/copy.h>
#include <ostream>

// this example illustrates how to concatenate two iterators
// examples:
//   concat_iterator([0, 1, 2, 3], [1, 1]) -> [0, 1, 2, 3, 1, 1]

template <typename Iterator1, typename Iterator2>
class concat_iterator
{
    public:

    typedef typename thrust::iterator_value<Iterator1>::type      value_type;
    typedef typename thrust::iterator_difference<Iterator1>::type difference_type;

    struct concat_select_functor : public thrust::unary_function<difference_type,value_type>
    {
        Iterator1 first;
        Iterator2 second;
        difference_type first_size;

        concat_select_functor(Iterator1 first, Iterator2 second, difference_type first_size)
            : first(first), second(second), first_size(first_size) {}

        __host__ __device__
        value_type operator()(const difference_type& i) const
        {
            return i < first_size ? first[i] : second[i-first_size];
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                          CountingIterator;
    typedef typename thrust::transform_iterator<concat_select_functor, CountingIterator> TransformIterator;

    // type of the concat_iterator
    typedef TransformIterator iterator;

    // construct concat_iterator using first_begin and second_begin
    concat_iterator(Iterator1 first_begin, Iterator1 first_end, Iterator2 second_begin, Iterator2 second_end)
        : first_begin(first_begin), first_end(first_end), second_begin(second_begin), second_end(second_end) {}

    iterator begin(void) const
    {
        return TransformIterator(CountingIterator(0), concat_select_functor(first_begin, second_begin, first_end-first_begin));
    }

    iterator end(void) const
    {
        return begin() + (first_end-first_begin) + (second_end-second_begin);
    }

    protected:
    Iterator1 first_begin;
    Iterator1 first_end;
    Iterator2 second_begin;
    Iterator2 second_end;
};

int main(void)
{
    typedef thrust::counting_iterator<int> CountingIterator;
    typedef thrust::constant_iterator<int> ConstantIterator;

    // create concat_iterator that combines a counting_iterator and constant_iterator
    concat_iterator<CountingIterator,ConstantIterator> vals(CountingIterator(0), CountingIterator(6), ConstantIterator(10), ConstantIterator(10) + 4);

    std::cout << "concat: ";
    thrust::copy(vals.begin(), vals.end(), std::ostream_iterator<int>(std::cout, " "));  std::cout << std::endl;

    return 0;
}
