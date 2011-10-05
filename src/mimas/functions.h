#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <boost/array.hpp>
#include <cmath>
#include <complex>
#include <functional>
#include "mimasexception.h"

namespace mimas {

#ifndef sgn
#define sgn(x) ((x<0)?-1:((x>0)?1:0))
#endif

/// Absolute value.
template< typename T >
struct _abs: public std::unary_function< T, T > 
{
  /** Compute absolute value.
      @param x A number.
      @return Absolute value of \c x. */
  T operator()( const T &x ) const { return std::abs( x ); }
};

/** Fast square.
    Compute square of values between -511 and +511 using a precomputed
    table. */
template< typename T >
struct _fastsqr: public std::unary_function< T, int >
{
  /// Constructor.
  _fastsqr(void) {
    for ( int i=0; i<(signed)table.size(); i++ )
      table[i] = i * i;
  }
  /// Function.
  int operator()( const T &x ) const {
    assert( x > -(signed)table.size() && x < (signed)table.size() );
    return table[ x < 0 ? -x : x ];
  }
  /// Table with precomputed values.
  boost::array< int, 512 > table;
};

/// Square.
template< typename T >
struct _sqr: public std::unary_function< T, T > 
{
  T operator()( const T &x ) const { return x * x; }
};

/// Square root.
template<typename T>
struct _sqrt: public std::unary_function< T, T >
{
  /** Compute square root.
      @param x A number.
      @return Square root of \c x. */
  T operator()( const T &x ) const { return (T)std::sqrt( (float)x ); }
};

/// Thresholding function.
template< typename T >
struct _threshold: public std::binary_function< T, T, T >
{
  /** Compare value with threshold.
      @param x The value to be considered.
      @param y The threshold to compare with.
      @return Default-value, if \c x is lower than \c y. Value of \c x
      otherwise. */
  T operator()( const T &x, const T &y ) const { return x < y ? T() : x; }
};

/// Thresholding function.
template< typename T >
struct _tobinary: public std::binary_function< T, T, bool >
{
  /** Compare value with threshold.
      @param x The value to be considered.
      @param y The threshold to compare with.
      @return A boolean, which is indicating, wether \c x is greater or equal
      to \c y. */
  bool operator()( const T &x, const T &y ) const { return x >= y; }
};


/// Convert boolean-pixel to bilevel-pixel.
template< typename T >
struct _bilevel: public std::binary_function< T, T, T > {
  _bilevel( T _val1, T _val2 ): val1(_val1), val2(_val2) {}
  /** Compare value with threshold and map to {val1,val2}.
      @param x The value to be considered.
      @param y The threshold to compare with.
      @return Bilevel-pixel, which is either \c val1 or \c val2. */
  T operator()( const T &x, const T &y ) const { return x >= y ? val2 : val1; }
  T val1;
  T val2;
};

/// Thresholding function with 2 levels
template< typename T >
struct _bilevel_double: public std::unary_function< T, T> {
  T val1, val2, min, max;
  _bilevel_double( T _val1, T _val2, T _min, T _max ): val1(_val1), val2(_val2), min(_min), max(_max){}
  /** Compare value with threshold levels . If value is between min
      and max, the output is val2, else it's val1.
      @param x The value to be considered.
      @return Bilevel-pixel, which is either \c val1 or \c val2. */
  T operator()( const T &x ) const { return (x >= min && x <= max) ? val2 : val1; 
  }
};

/// Linear companding function.    
template< typename T >
struct _normalise: public std::unary_function< T, T >
{
  ///
  _normalise( T _minval, T _maxval, T _val1, T _val2 ) {
    if ( _maxval > _minval ) {
      factor = (double)( _val2 - _val1 ) / ( _maxval - _minval );
      offset = _val1 - _minval * factor;
    } else {
      factor = 0.0;
      offset = _val1;
    };
  }
  T operator()( const T &x ) const {
    // Scale each pixel-value: ( pixel - minval ) * factor + val1.
    // pixel * factor - minval * factor + val1
    return (T)( x * factor + offset );
  }
  double factor;
  double offset;
};

/// Take norm of a real or complex value.
template< typename T1, typename T2 >
struct _norm: public std::unary_function< T2, T1 >
{
  T1 operator()( const T2 &x ) const { return std::norm( x ); }
};

/// The argument of a complex value.
template< typename T1, typename T2 >
struct _arg: public std::unary_function< T2, T1 >
{
  T1 operator()( const T2 &x ) const { return std::arg( x ); }
};

/// Complex conjugate.
template< typename T >
struct _conj: public std::unary_function< T, T >
{
  T operator()( const T &x ) const { return std::conj( x ); }
};

/// Compute logarithm.
template< typename T >
struct _log: public std::unary_function< T, T >
{
  T operator()( const T &x ) const { return log( x ); }
};

/** Compute sum of squares.
    The sum of squares can be computed with the multiplication- and
    plus-operator as well, but it would require allocation of one temporary
    array. */
template< typename T >
struct _sumsquares: public std::binary_function< T, T, T > {
  /** Compute sum of squares.
      @param x First value.
      @param y Second value.
      @return x^2+y^2 */
  T operator()( const T &x, const T &y ) const { return x * x + y * y; }
};

/// Compute angle.
template< typename T >
struct _orientation: public std::binary_function< T, T, T > {
  /** Compute sum of squares.
      @param y y-component
      @param x x-component
      @return atan2( y, x ) */
  T operator()( const T &y, const T &x ) const { return atan2( y, x ); }
};

};

#endif
