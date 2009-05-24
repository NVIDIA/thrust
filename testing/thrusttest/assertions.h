#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/iterator_traits.h>

#include <thrusttest/exceptions.h>
#include <thrusttest/util.h>

#define ASSERT_EQUAL_QUIET(X,Y)  thrusttest::assert_equal_quiet((X),(Y), __FILE__, __LINE__)
#define ASSERT_EQUAL(X,Y)        thrusttest::assert_equal((X),(Y), __FILE__,  __LINE__)
#define ASSERT_LEQUAL(X,Y)       thrusttest::assert_lequal((X),(Y), __FILE__,  __LINE__)
#define ASSERT_GEQUAL(X,Y)       thrusttest::assert_gequal((X),(Y), __FILE__,  __LINE__)
#define ASSERT_ALMOST_EQUAL(X,Y) thrusttest::assert_almost_equal((X),(Y), __FILE__, __LINE__)
#define KNOWN_FAILURE            { thrusttest::UnitTestKnownFailure f; f << "[" << __FILE__ ":" << __LINE__ << "]"; throw f;}


namespace thrusttest
{

static size_t MAX_OUTPUT_LINES = 10;

static double DEFAULT_RELATIVE_TOL = 1e-4;
static double DEFAULT_ABSOLUTE_TOL = 1e-4;

////
// check scalar values
template <typename T1, typename T2>
void assert_equal(const T1& a, const T2& b, 
                  const std::string& filename = "unknown", int lineno = -1)
{
    if(!(a == b)){
        thrusttest::UnitTestFailure f;
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
        thrusttest::UnitTestFailure f;
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
        thrusttest::UnitTestFailure f;
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
    if(!(a >= b)){
        thrusttest::UnitTestFailure f;
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
        thrusttest::UnitTestFailure f;
        f << "[" << filename << ":" << lineno << "] ";
        f << "values are not approximately equal: " << (double) a << " " << (double) b;
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
            return almost_equal(a, b, a_tol, r_tol);
        }
};

////
// check sequences

template <typename ForwardIterator, typename BinaryPredicate>
void assert_equal(ForwardIterator first1, ForwardIterator last1, ForwardIterator first2, const BinaryPredicate& op,
                  const std::string& filename = "unknown", int lineno = -1)
{
    size_t i = 0;
    size_t mismatches = 0;
    
    typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;

    thrusttest::UnitTestFailure f;
    f << "[" << filename << ":" << lineno << "] ";
    f << "Sequences are not equal [type='" << type_name<InputType>() << "']\n";
    f << "--------------------------------\n";

    while(first1 != last1){
        if(!op(*first1, *first2)){
            mismatches++;
            if(mismatches <= MAX_OUTPUT_LINES)
                f << "  [" << i << "] " << *first1 << "  " << *first2 << "\n";
        }

        first1++;
        first2++;
        i++;
    }


    if (mismatches > 0){
        if(mismatches > MAX_OUTPUT_LINES)
            f << "  (output limit reached)\n";
        f << "--------------------------------\n";
        f << "Sequences differ at " << mismatches << " of " << i << " positions" << "\n";
        throw f;
    }

}

template <typename ForwardIterator>
void assert_equal(ForwardIterator first1, ForwardIterator last1, ForwardIterator first2,
                  const std::string& filename = "unknown", int lineno = -1)
{
    typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;
    assert_equal(first1, last1, first2, thrust::equal_to<InputType>(), filename, lineno);
}


template <typename ForwardIterator>
void assert_almost_equal(ForwardIterator first1, ForwardIterator last1, ForwardIterator first2, 
                         const std::string& filename = "unknown", int lineno = -1,
                         const double a_tol = DEFAULT_ABSOLUTE_TOL, const double r_tol = DEFAULT_RELATIVE_TOL)
{
    typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;
    assert_equal(first1, last1, first2, almost_equal_to<InputType>(a_tol, r_tol), filename, lineno);
}


template <typename T>
void assert_equal(const thrust::host_vector<T>& A, const thrust::host_vector<T>& B,
                  const std::string& filename = "unknown", int lineno = -1)
{
    if(A.size() != B.size())
        throw thrusttest::UnitTestError("Sequences have different sizes");
    assert_equal(A.begin(), A.end(), B.begin(), filename, lineno);
}

template <typename T>
void assert_almost_equal(const thrust::host_vector<T>& A, const thrust::host_vector<T>& B, 
                         const std::string& filename = "unknown", int lineno = -1,
                         const double a_tol = DEFAULT_ABSOLUTE_TOL, const double r_tol = DEFAULT_RELATIVE_TOL)
{
    if(A.size() != B.size())
        throw thrusttest::UnitTestError("Sequences have different sizes");
    assert_almost_equal(A.begin(), A.end(), B.begin(), filename, lineno, a_tol, r_tol);
}

template <typename T>
void assert_equal(const thrust::host_vector<T>& A, const thrust::device_vector<T>& B,
                  const std::string& filename = "unknown", int lineno = -1)
{
    thrust::host_vector<T> B_host = B;
    assert_equal(A, B_host, filename, lineno);
}

template <typename T>
void assert_equal(const thrust::device_vector<T>& A, const thrust::host_vector<T>& B,
                  const std::string& filename = "unknown", int lineno = -1)
{
    thrust::host_vector<T> A_host = A;
    assert_equal(A_host, B, filename, lineno);
}

template <typename T>
void assert_equal(const thrust::device_vector<T>& A, const thrust::device_vector<T>& B,
                  const std::string& filename = "unknown", int lineno = -1)
{
    thrust::host_vector<T> A_host = A;
    thrust::host_vector<T> B_host = B;
    assert_equal(A_host, B_host, filename, lineno);
}

template <typename T>
void assert_almost_equal(const thrust::host_vector<T>& A, const thrust::device_vector<T>& B,
                         const std::string& filename = "unknown", int lineno = -1,
                         const double a_tol = DEFAULT_ABSOLUTE_TOL, const double r_tol = DEFAULT_RELATIVE_TOL)
{
    thrust::host_vector<T> B_host = B;
    assert_almost_equal(A, B_host, filename, lineno, a_tol, r_tol);
}

template <typename T>
void assert_almost_equal(const thrust::device_vector<T>& A, const thrust::host_vector<T>& B,
                         const std::string& filename = "unknown", int lineno = -1,
                         const double a_tol = DEFAULT_ABSOLUTE_TOL, const double r_tol = DEFAULT_RELATIVE_TOL)
{
    thrust::host_vector<T> A_host = A;
    assert_almost_equal(A_host, B, filename, lineno, a_tol, r_tol);
}

template <typename T>
void assert_almost_equal(const thrust::device_vector<T>& A, const thrust::device_vector<T>& B,
                         const std::string& filename = "unknown", int lineno = -1,
                         const double a_tol = DEFAULT_ABSOLUTE_TOL, const double r_tol = DEFAULT_RELATIVE_TOL)
{
    thrust::host_vector<T> A_host = A;
    thrust::host_vector<T> B_host = B;
    assert_almost_equal(A_host, B_host, filename, lineno, a_tol, r_tol);
}

}; //end namespace thrusttest
