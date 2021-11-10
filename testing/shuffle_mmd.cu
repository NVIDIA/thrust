#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2011
#include <map>
#include <limits>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <unittest/unittest.h>

// Inverse error function
// https://github.com/lakshayg/erfinv
/*
MIT License
Copyright (c) 2017-2019 Lakshay Garg <lakshayg@outlook.in>
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

long double erfinv( long double x )
{

    if( x < -1 || x > 1 )
    {
        return NAN;
    }
    else if( x == 1.0 )
    {
        return INFINITY;
    }
    else if( x == -1.0 )
    {
        return -INFINITY;
    }

    const long double LN2 = 6.931471805599453094172321214581e-1L;

    const long double A0 = 1.1975323115670912564578e0L;
    const long double A1 = 4.7072688112383978012285e1L;
    const long double A2 = 6.9706266534389598238465e2L;
    const long double A3 = 4.8548868893843886794648e3L;
    const long double A4 = 1.6235862515167575384252e4L;
    const long double A5 = 2.3782041382114385731252e4L;
    const long double A6 = 1.1819493347062294404278e4L;
    const long double A7 = 8.8709406962545514830200e2L;

    const long double B0 = 1.0000000000000000000e0L;
    const long double B1 = 4.2313330701600911252e1L;
    const long double B2 = 6.8718700749205790830e2L;
    const long double B3 = 5.3941960214247511077e3L;
    const long double B4 = 2.1213794301586595867e4L;
    const long double B5 = 3.9307895800092710610e4L;
    const long double B6 = 2.8729085735721942674e4L;
    const long double B7 = 5.2264952788528545610e3L;

    const long double C0 = 1.42343711074968357734e0L;
    const long double C1 = 4.63033784615654529590e0L;
    const long double C2 = 5.76949722146069140550e0L;
    const long double C3 = 3.64784832476320460504e0L;
    const long double C4 = 1.27045825245236838258e0L;
    const long double C5 = 2.41780725177450611770e-1L;
    const long double C6 = 2.27238449892691845833e-2L;
    const long double C7 = 7.74545014278341407640e-4L;

    const long double D0 = 1.4142135623730950488016887e0L;
    const long double D1 = 2.9036514445419946173133295e0L;
    const long double D2 = 2.3707661626024532365971225e0L;
    const long double D3 = 9.7547832001787427186894837e-1L;
    const long double D4 = 2.0945065210512749128288442e-1L;
    const long double D5 = 2.1494160384252876777097297e-2L;
    const long double D6 = 7.7441459065157709165577218e-4L;
    const long double D7 = 1.4859850019840355905497876e-9L;

    const long double E0 = 6.65790464350110377720e0L;
    const long double E1 = 5.46378491116411436990e0L;
    const long double E2 = 1.78482653991729133580e0L;
    const long double E3 = 2.96560571828504891230e-1L;
    const long double E4 = 2.65321895265761230930e-2L;
    const long double E5 = 1.24266094738807843860e-3L;
    const long double E6 = 2.71155556874348757815e-5L;
    const long double E7 = 2.01033439929228813265e-7L;

    const long double F0 = 1.414213562373095048801689e0L;
    const long double F1 = 8.482908416595164588112026e-1L;
    const long double F2 = 1.936480946950659106176712e-1L;
    const long double F3 = 2.103693768272068968719679e-2L;
    const long double F4 = 1.112800997078859844711555e-3L;
    const long double F5 = 2.611088405080593625138020e-5L;
    const long double F6 = 2.010321207683943062279931e-7L;
    const long double F7 = 2.891024605872965461538222e-15L;

    long double abs_x = fabsl( x );

    if( abs_x <= 0.85L )
    {
        long double r = 0.180625L - 0.25L * x * x;
        long double num =
            ( ( ( ( ( ( ( A7 * r + A6 ) * r + A5 ) * r + A4 ) * r + A3 ) * r + A2 ) * r + A1 ) * r + A0 );
        long double den =
            ( ( ( ( ( ( ( B7 * r + B6 ) * r + B5 ) * r + B4 ) * r + B3 ) * r + B2 ) * r + B1 ) * r + B0 );
        return x * num / den;
    }

    long double r = sqrtl( LN2 - logl( 1.0L - abs_x ) );

    long double num, den;
    if( r <= 5.0L )
    {
        r = r - 1.6L;
        num = ( ( ( ( ( ( ( C7 * r + C6 ) * r + C5 ) * r + C4 ) * r + C3 ) * r + C2 ) * r + C1 ) * r + C0 );
        den = ( ( ( ( ( ( ( D7 * r + D6 ) * r + D5 ) * r + D4 ) * r + D3 ) * r + D2 ) * r + D1 ) * r + D0 );
    }
    else
    {
        r = r - 5.0L;
        num = ( ( ( ( ( ( ( E7 * r + E6 ) * r + E5 ) * r + E4 ) * r + E3 ) * r + E2 ) * r + E1 ) * r + E0 );
        den = ( ( ( ( ( ( ( F7 * r + F6 ) * r + F5 ) * r + F4 ) * r + F3 ) * r + F2 ) * r + F1 ) * r + F0 );
    }

    return copysignl( num / den, x );
}

long double erfinv_refine( long double x, int nr_iter )
{
    const long double k = 0.8862269254527580136490837416706L; // 0.5 * sqrt(pi)
    long double y = erfinv( x );
    while( nr_iter-- > 0 )
    {
        y -= k * ( erfl( y ) - x ) / expl( -y * y );
    }
    return y;
}

#define LSBIT( i ) ( ( i ) & -( i ) )

class FenwickTree
{
    std::vector<size_t> data;

public:
    FenwickTree( size_t n ) : data( n )
    {
    }
    void Add( size_t i )
    {
        for( ; i < data.size(); i += LSBIT( i + 1 ) )
        {
            data[i]++;
        }
    }
    int GetCount( size_t i )
    {
        int sum = 0;
        for( ; i > 0; i -= LSBIT( i ) )
            sum += data[i - 1];
        return sum;
    }
};

template <typename Vector>
size_t ConcordantPairs( const Vector& x )
{
    size_t count = 0;
    FenwickTree tree( x.size() );
    for( auto x_i : x )
    {
        count += tree.GetCount( x_i );
        tree.Add( x_i );
    }
    return count;
}

template <typename Vector>
double MallowsKernelIdentity( const Vector& x, double lambda )
{
    auto con = ConcordantPairs( x );
    auto norm = x.size() * ( x.size() - 1 ) / 2;
    double y = 1 - ( double( con ) / norm );
    return exp( -lambda * y );
}

double MallowsExpectedValue( size_t n, double lambda )
{
    double norm = n * ( n - 1 ) / 2.0;
    double product = 1.0;
    for( size_t j = 1; j <= n; j++ )
    {
        product *= ( 1.0 - exp( -lambda * j / norm ) ) / ( j * ( 1.0 - exp( -lambda / norm ) ) );
    }
    return product;
}

double HoeffdingAcceptanceThreshold( double alpha, size_t num_samples )
{
    double w = log( 2 / alpha ) / ( 2 * num_samples );
    return sqrt( w );
}

double NormalAcceptanceThreshold( double alpha, size_t num_samples, size_t n, double lambda )
{
    double var = (MallowsExpectedValue( n, 2 * lambda ) - pow( MallowsExpectedValue( n, lambda ), 2.0 )) / num_samples;
    return sqrt( 2 * var ) * erfinv_refine( 1 - alpha, 10 );
}

template <typename Vector>
void TestShuffleMallows() {
  typedef typename Vector::value_type T;

  const uint32_t shuffle_size = std::min((uint32_t)(1u << 13) + 1, (uint32_t)std::numeric_limits<T>::max());
  const uint32_t num_samples = 1000;
  const double lambda = 5;

  thrust::default_random_engine g(0xD5);
  Vector sequence(shuffle_size);
  double mallows_expected = 0;
  for( uint32_t i = 0; i < num_samples; i++ )
  {
      thrust::sequence(sequence.begin(), sequence.end(), 0);
      thrust::shuffle(sequence.begin(), sequence.end(), g);

      thrust::host_vector<T> tmp(sequence.begin(), sequence.end());
      mallows_expected += MallowsKernelIdentity( tmp, lambda );
  }

  mallows_expected /= num_samples;
  double mmd = abs( mallows_expected - MallowsExpectedValue( shuffle_size, lambda ) );

  const double alpha = 0.01;
  ASSERT_LESS(mmd, HoeffdingAcceptanceThreshold( alpha, num_samples ));
  ASSERT_LESS(mmd, NormalAcceptanceThreshold( alpha, num_samples, shuffle_size, lambda ));

}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestShuffleMallows);


#endif
