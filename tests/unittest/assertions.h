#pragma once

#include <thrust/complex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>

#include <unittest/exceptions.h>
#include <unittest/util.h>

#define ASSERT_EQUAL_QUIET(X,Y)  unittest::assert_equal_quiet((X),(Y), __FILE__, __LINE__)
#define ASSERT_EQUAL(X,Y)        unittest::assert_equal((X),(Y), __FILE__,  __LINE__)
#define ASSERT_LEQUAL(X,Y)       unittest::assert_lequal((X),(Y), __FILE__,  __LINE__)
#define ASSERT_GEQUAL(X,Y)       unittest::assert_gequal((X),(Y), __FILE__,  __LINE__)
#define ASSERT_ALMOST_EQUAL(X,Y) unittest::assert_almost_equal((X),(Y), __FILE__, __LINE__)
#define KNOWN_FAILURE            { unittest::UnitTestKnownFailure f; f << "[" << __FILE__ ":" << __LINE__ << "]"; throw f;}
                    
#define ASSERT_EQUAL_RANGES(X,Y,Z)  unittest::assert_equal((X),(Y),(Z), __FILE__,  __LINE__)

#define ASSERT_THROWS(X,Y)                                                         \
    {   bool thrown = false; try { X; } catch (Y) { thrown = true; }                  \
        if (!thrown) { unittest::UnitTestFailure f; f << "[" << __FILE__ << ":" << __LINE__ << "] did not throw " << #Y; throw f; } \
    }


namespace unittest
{

static size_t MAX_OUTPUT_LINES = 10;

static double DEFAULT_RELATIVE_TOL = 1e-4;
static double DEFAULT_ABSOLUTE_TOL = 1e-4;

template<typename T>
  struct value_type
{
  typedef typename thrust::detail::remove_const<
    typename thrust::detail::remove_reference<
      T
    >::type
  >::type type;
};

template<typename T>
  struct value_type< thrust::device_reference<T> >
{
  typedef typename value_type<T>::type type;
};

////
// check scalar values
template <typename T1, typename T2>
void assert_equal(const T1& a, const T2& b, 
                  const std::string& filename = "unknown", int lineno = -1)
{
    // convert a & b to a's value_type to avoid warning upon comparison
    typedef typename value_type<T1>::type T;

    if(!(T(a) == T(b))){
        unittest::UnitTestFailure f;
        f << "[" << filename << ":" << lineno << "] ";
        f << "values are not equal: " << a << " " << b;
        f << " [type='" << type_name<T1>() << "']";
        throw f;
    }
}

// sometimes it's not possible to << a type
template <typename T1, typename T2>
void assert_equal_quiet(const T1& a, const T2& b, 
                        const std::string& filename = "unknown", int lineno = -1)
{
    if(!(a == b)){
        unittest::UnitTestFailure f;
        f << "[" << filename << ":" << lineno << "] ";
        f << "values are not equal.";
        f << " [type='" << type_name<T1>() << "']";
        throw f;
    }
}

template <typename T1, typename T2>
void assert_lequal(const T1& a, const T2& b, 
                   const std::string& filename = "unknown", int lineno = -1)
{
    if(!(a <= b)){
        unittest::UnitTestFailure f;
        f << "[" << filename << ":" << lineno << "] ";
        f << a << " is greater than " << b;
        f << " [type='" << type_name<T1>() << "']";
        throw f;
    }
}

template <typename T1, typename T2>
void assert_gequal(const T1& a, const T2& b, 
                   const std::string& filename = "unknown", int lineno = -1)
{
    if(!(a >= T1(b))){
        unittest::UnitTestFailure f;
        f << "[" << filename << ":" << lineno << "] ";
        f << a << " is less than " << b;
        f << " [type='" << type_name<T1>() << "']";
        throw f;
    }
}

// define our own abs() because std::abs() isn't portable for all types for some reason
template<typename T>
  T abs(const T &x)
{
  return x > 0 ? x : -x;
}


inline
bool almost_equal(const double& a, const double& b, const double& a_tol, const double& r_tol)
{
    if(abs(a - b) > r_tol * (abs(a) + abs(b)) + a_tol)
        return false;
    else
        return true;
}

template <typename T1, typename T2>
void assert_almost_equal(const T1& a, const T2& b, 
                         const std::string& filename = "unknown", int lineno = -1,
                         double a_tol = DEFAULT_ABSOLUTE_TOL, double r_tol = DEFAULT_RELATIVE_TOL)

{
    if(!almost_equal(a, b, a_tol, r_tol)){
        unittest::UnitTestFailure f;
        f << "[" << filename << ":" << lineno << "] ";
        f << "values are not approximately equal: " << (double) a << " " << (double) b;
        f << " [type='" << type_name<T1>() << "']";
        throw f;
    }
}


template <typename T1, typename T2>
  void assert_almost_equal(const thrust::complex<T1>& a, const thrust::complex<T2>& b, 
                         const std::string& filename = "unknown", int lineno = -1,
                         double a_tol = DEFAULT_ABSOLUTE_TOL, double r_tol = DEFAULT_RELATIVE_TOL)

{
  if(!almost_equal(a.real(), b.real(), a_tol, r_tol)){
        unittest::UnitTestFailure f;
        f << "[" << filename << ":" << lineno << "] ";
        f << "values are not approximately equal: " <<  a << " " << b;
        f << " [type='" << type_name<T1>() << "']";
        throw f;
    }
}


template <typename T1, typename T2>
  void assert_almost_equal(const thrust::complex<T1>& a, const std::complex<T2>& b, 
                         const std::string& filename = "unknown", int lineno = -1,
                         double a_tol = DEFAULT_ABSOLUTE_TOL, double r_tol = DEFAULT_RELATIVE_TOL)

{
  if(!almost_equal(a.real(), b.real(), a_tol, r_tol)){
        unittest::UnitTestFailure f;
        f << "[" << filename << ":" << lineno << "] ";
        f << "values are not approximately equal: " <<  a << " " << b;
        f << " [type='" << type_name<T1>() << "']";
        throw f;
    }
}

template <typename T>
class almost_equal_to
{
    public:
        double a_tol, r_tol;
        almost_equal_to(double _a_tol = DEFAULT_ABSOLUTE_TOL, double _r_tol = DEFAULT_RELATIVE_TOL) : a_tol(_a_tol), r_tol(_r_tol) {}
        bool operator()(const T& a, const T& b) const {
            return almost_equal((double) a, (double) b, a_tol, r_tol);
        }
};


template <typename T>
class almost_equal_to<thrust::complex<T> >
{
    public:
        double a_tol, r_tol;
        almost_equal_to(double _a_tol = DEFAULT_ABSOLUTE_TOL, double _r_tol = DEFAULT_RELATIVE_TOL) : a_tol(_a_tol), r_tol(_r_tol) {}
        bool operator()(const thrust::complex<T>& a, const thrust::complex<T>& b) const {
	  return almost_equal((double) a.real(), (double) b.real(), a_tol, r_tol) && 
	    almost_equal((double) a.imag(), (double) b.imag(), a_tol, r_tol);
        }
};

////
// check sequences

template <typename ForwardIterator1, typename ForwardIterator2, typename BinaryPredicate>
void assert_equal(ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2, BinaryPredicate op,
                  const std::string& filename = "unknown", int lineno = -1)
{
    typedef typename thrust::iterator_difference<ForwardIterator1>::type difference_type;
    typedef typename thrust::iterator_value<ForwardIterator1>::type InputType;
    
    bool failure = false;

    difference_type length1 = thrust::distance(first1, last1);
    difference_type length2 = thrust::distance(first2, last2);
    
    difference_type min_length = thrust::min(length1, length2);

    unittest::UnitTestFailure f;
    f << "[" << filename << ":" << lineno << "] ";

    // check lengths
    if (length1 != length2)
    {
      failure = true;
      f << "Sequences have different sizes (" << length1 << " != " << length2 << ")\n";
    }

    // check values
    
    size_t mismatches = 0;

    for (difference_type i = 0; i < min_length; i++)
    {
      if(!op(*first1, *first2))
      {
        if (mismatches == 0)
        {
          failure = true;
          f << "Sequences are not equal [type='" << type_name<InputType>() << "']\n";
          f << "--------------------------------\n";
        }

        mismatches++;

        if(mismatches <= MAX_OUTPUT_LINES)
        {
          if (sizeof(InputType) == 1)
            f << "  [" << i << "] " << *first1 + InputType() << "  " << *first2 + InputType() << "\n"; // unprintable chars are a problem
          else
            f << "  [" << i << "] " << *first1 << "  " << *first2 << "\n";
        }
      }

      first1++;
      first2++;
    }

    if (mismatches > 0)
    {
      if(mismatches > MAX_OUTPUT_LINES)
          f << "  (output limit reached)\n";
      f << "--------------------------------\n";
      f << "Sequences differ at " << mismatches << " of " << min_length << " positions" << "\n";
    }
    else if (length1 != length2)
    {
      f << "Sequences agree through " << min_length << " positions [type='" << type_name<InputType>() << "']\n";
    }

    if (failure)
      throw f;
}

template <typename ForwardIterator1, typename ForwardIterator2>
void assert_equal(ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2,
                  const std::string& filename = "unknown", int lineno = -1)
{
    typedef typename thrust::iterator_traits<ForwardIterator1>::value_type InputType;
    assert_equal(first1, last1, first2, last2, thrust::equal_to<InputType>(), filename, lineno);
}


template <typename ForwardIterator1, typename ForwardIterator2>
void assert_almost_equal(ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2,
                         const std::string& filename = "unknown", int lineno = -1,
                         const double a_tol = DEFAULT_ABSOLUTE_TOL, const double r_tol = DEFAULT_RELATIVE_TOL)
{
    typedef typename thrust::iterator_traits<ForwardIterator1>::value_type InputType;
    assert_equal(first1, last1, first2, last2, almost_equal_to<InputType>(a_tol, r_tol), filename, lineno);
}


template <typename T, typename Alloc>
void assert_equal(const thrust::host_vector<T,Alloc>& A, const thrust::host_vector<T,Alloc>& B,
                  const std::string& filename = "unknown", int lineno = -1)
{
    assert_equal(A.begin(), A.end(), B.begin(), B.end(), filename, lineno);
}

template <typename T, typename Alloc>
void assert_almost_equal(const thrust::host_vector<T,Alloc>& A, const thrust::host_vector<T,Alloc>& B, 
                         const std::string& filename = "unknown", int lineno = -1,
                         const double a_tol = DEFAULT_ABSOLUTE_TOL, const double r_tol = DEFAULT_RELATIVE_TOL)
{
    assert_almost_equal(A.begin(), A.end(), B.begin(), B.end(), filename, lineno, a_tol, r_tol);
}

template <typename T, typename Alloc1, typename Alloc2>
void assert_equal(const thrust::host_vector<T,Alloc1>& A, const thrust::device_vector<T,Alloc2>& B,
                  const std::string& filename = "unknown", int lineno = -1)
{
    thrust::host_vector<T,Alloc1> B_host = B;
    assert_equal(A, B_host, filename, lineno);
}

template <typename T, typename Alloc1, typename Alloc2>
void assert_equal(const thrust::device_vector<T,Alloc1>& A, const thrust::host_vector<T,Alloc2>& B,
                  const std::string& filename = "unknown", int lineno = -1)
{
    thrust::host_vector<T,Alloc2> A_host = A;
    assert_equal(A_host, B, filename, lineno);
}

template <typename T, typename Alloc>
void assert_equal(const thrust::device_vector<T,Alloc>& A, const thrust::device_vector<T,Alloc>& B,
                  const std::string& filename = "unknown", int lineno = -1)
{
    thrust::host_vector<T> A_host = A;
    thrust::host_vector<T> B_host = B;
    assert_equal(A_host, B_host, filename, lineno);
}

template <typename T, typename Alloc1, typename Alloc2>
void assert_almost_equal(const thrust::host_vector<T,Alloc1>& A, const thrust::device_vector<T,Alloc2>& B,
                         const std::string& filename = "unknown", int lineno = -1,
                         const double a_tol = DEFAULT_ABSOLUTE_TOL, const double r_tol = DEFAULT_RELATIVE_TOL)
{
    thrust::host_vector<T,Alloc1> B_host = B;
    assert_almost_equal(A, B_host, filename, lineno, a_tol, r_tol);
}

template <typename T, typename Alloc1, typename Alloc2>
void assert_almost_equal(const thrust::device_vector<T,Alloc1>& A, const thrust::host_vector<T,Alloc2>& B,
                         const std::string& filename = "unknown", int lineno = -1,
                         const double a_tol = DEFAULT_ABSOLUTE_TOL, const double r_tol = DEFAULT_RELATIVE_TOL)
{
    thrust::host_vector<T,Alloc2> A_host = A;
    assert_almost_equal(A_host, B, filename, lineno, a_tol, r_tol);
}

template <typename T, typename Alloc>
void assert_almost_equal(const thrust::device_vector<T,Alloc>& A, const thrust::device_vector<T,Alloc>& B,
                         const std::string& filename = "unknown", int lineno = -1,
                         const double a_tol = DEFAULT_ABSOLUTE_TOL, const double r_tol = DEFAULT_RELATIVE_TOL)
{
    thrust::host_vector<T> A_host = A;
    thrust::host_vector<T> B_host = B;
    assert_almost_equal(A_host, B_host, filename, lineno, a_tol, r_tol);
}

}; //end namespace unittest
