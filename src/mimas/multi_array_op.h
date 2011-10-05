#ifndef __MULTI_ARRAY_OP_H
#define __MULTI_ARRAY_OP_H

#include <boost/array.hpp>
#include <boost/multi_array.hpp>
#include <functional>
#include "functions.h"

namespace mimas {
  /** @defgroup arrayOp Operators for boost::multi_array and mimas::image
      Mimas has support for doing element-wise operations and convolutions on
      \c boost::multi_array as well as mimas::image.

      Implementations for element-wise operations on multi-dimensional arrays
      are provided. \c boost::multi_array is used to represent the
      multidimensional arrays.

      The supported operations are:
      \li Associative array-scalar operators: multiplication, plus and minus.
      \li Non-associative array-scalar operators: division, threshold.
      \li Array-array operators: multiplication, division, plus and minus.

      The following example demonstrates, how versatile the array-operators
      provided by mimas are:
      \include arrayop/main.cc


      @author Jan Wedekind <jan@wedesoft.de> 
      @author Haifeng Gong <hfgong at users.sourceforge.net>
      @date Thu Jul 06 14:13:05 UTC 2006
      @todo Does not compile under Solaris at the moment.
      @todo Add documentation of all available array operators.
      @todo Define boost-mpl templates for mimas::rgba to allow the use of
      expression templates (how?)
      @{ */
  template<
    typename T1, typename T2, size_t NumDims, typename Allocator,
    template< typename, size_t, typename > class MultiArray
  >
  boost::multi_array< T1, NumDims > empty_clone
  ( const MultiArray< T2, NumDims, Allocator > &x )
  {
    boost::array< size_t, NumDims > shape;
    std::copy( x.shape(), x.shape() + NumDims, shape.begin() );
    boost::multi_array< T1, NumDims > retVal( shape );
    return retVal;
  }

  ///
  template<
    typename T, class F
    >
  boost::detail::multi_array::sub_array< T, 1 > multi_apply
  ( boost::detail::multi_array::sub_array< T, 1 > a, F f ) {
    for ( typename boost::detail::multi_array::sub_array< T, 1 >::iterator i = a.begin();
          i != a.end(); i++ )
      f( *i );
    return a;
  }

  ///
  template<
    typename T, class F,
    template< typename, size_t > class MultiArray
  >
  MultiArray< T, 1 > &multi_apply
  ( MultiArray< T, 1 > &a, F f ) {
    for ( typename MultiArray< T, 1 >::iterator i = a.begin();
          i != a.end(); i++ )
      f( *i );
    return a;
  }

#if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 2))
  ///
  template<
    typename T, class F, typename Allocator,
    template< typename, size_t, typename > class MultiArray
  >
  MultiArray< T, 1, Allocator > &multi_apply
  ( MultiArray< T, 1, Allocator > &a, F f ) {
    for ( typename MultiArray< T, 1, Allocator >::iterator i = a.begin();
          i != a.end(); i++ )
      f( *i );
    return a;
  }
#endif

  ///
  template<
    typename T, size_t NumDims, class F
    >
  boost::detail::multi_array::sub_array< T, NumDims > multi_apply
  ( boost::detail::multi_array::sub_array< T, NumDims > a, F f ) {
    for ( typename boost::detail::multi_array::sub_array< T, NumDims >::iterator i = a.begin();
          i != a.end(); i++ )
      multi_apply( *i, f );
    return a;
  }

  ///
  template<
    typename T, size_t NumDims, class F,
    template< typename, size_t > class MultiArray
  >
  MultiArray< T, NumDims > &multi_apply
  ( MultiArray< T, NumDims > &a, F f ) {
    for ( typename MultiArray< T, NumDims >::iterator i = a.begin();
          i != a.end(); i++ )
      multi_apply( *i, f );
    return a;
  }

#if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 2))
  ///
  template<
    typename T, size_t NumDims, class F, typename Allocator,
    template< typename, size_t, typename > class MultiArray
  >
  MultiArray< T, NumDims, Allocator > &multi_apply
  ( MultiArray< T, NumDims, Allocator > &a, F f ) {
    for ( typename MultiArray< T, NumDims, Allocator >::iterator i = a.begin();
          i != a.end(); i++ )
      multi_apply( *i, f );
    return a;
  }
#endif

  ///
  template<
    typename T1, typename T2, class F, typename Allocator,
    template< typename, size_t, typename > class MultiArray
  >
  boost::detail::multi_array::sub_array< T1, 1 > multi_apply
  ( boost::detail::multi_array::sub_array< T1, 1 > a,
    const MultiArray< T2, 1, Allocator > &b,
    F f ) {
    typename MultiArray< T2, 1, Allocator >::const_iterator j = b.begin();
    for ( typename boost::detail::multi_array::sub_array< T1, 1 >::iterator i = a.begin();
          i != a.end(); i++, j++ )
      f( *i, *j );
    return a;
  }

  ///
  template<
    typename T1, typename T2, class F,
    typename Allocator2,
    template< typename, size_t > class MultiArray1,
    template< typename, size_t, typename > class MultiArray2
  >
  MultiArray1< T1, 1 > &multi_apply
  ( MultiArray1< T1, 1 > &a,
    const MultiArray2< T2, 1, Allocator2 > &b, F f ) {
    typename MultiArray2< T2, 1, Allocator2 >::const_iterator j = b.begin();
    for ( typename MultiArray1< T1, 1 >::iterator i = a.begin();
          i != a.end(); i++, j++ )
      f( *i, *j );
    return a;
  }

#if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 2))
  ///
  template<
    typename T1, typename T2, class F,
    typename Allocator1, typename Allocator2,
    template< typename, size_t, typename > class MultiArray1,
    template< typename, size_t, typename > class MultiArray2
  >
  MultiArray1< T1, 1, Allocator1 > &multi_apply
  ( MultiArray1< T1, 1, Allocator1 > &a,
    const MultiArray2< T2, 1, Allocator2 > &b, F f ) {
    typename MultiArray2< T2, 1, Allocator2 >::const_iterator j = b.begin();
    for ( typename MultiArray1< T1, 1, Allocator1 >::iterator i = a.begin();
          i != a.end(); i++, j++ )
      f( *i, *j );
    return a;
  }
#endif

  ///
  template<
    typename T1, typename T2, size_t NumDims, class F, typename Allocator,
    template< typename, size_t, typename > class MultiArray
  >
  boost::detail::multi_array::sub_array< T1, NumDims > multi_apply
  ( boost::detail::multi_array::sub_array< T1, NumDims > a,
    const MultiArray< T2, NumDims, Allocator > &b,
    F f ) {
    typename MultiArray< T2, NumDims, Allocator >::const_iterator j = b.begin();
    for ( typename boost::detail::multi_array::sub_array< T1, NumDims >::iterator i = a.begin();
          i != a.end(); i++, j++ )
      multi_apply( *i, *j, f );
    return a;
  }

  ///
  template<
    typename T1, typename T2, size_t NumDims, class F
  >
  boost::detail::multi_array::sub_array< T1, NumDims > multi_apply
  ( boost::detail::multi_array::sub_array< T1, NumDims > a,
    const boost::detail::multi_array::sub_array< T2, NumDims > b,
    F f ) {
    typename boost::detail::multi_array::sub_array< T2, NumDims >::const_iterator j = b.begin();
    for ( typename boost::detail::multi_array::sub_array< T1, NumDims >::iterator i = a.begin();
          i != a.end(); i++, j++ )
      multi_apply( *i, *j, f );
    return a;
  }

  ///
  template<
    typename T1, typename T2, size_t NumDims, class F,
    typename Allocator2,
    template< typename, size_t > class MultiArray1,
    template< typename, size_t, typename > class MultiArray2
  >
  MultiArray1< T1, NumDims > &multi_apply
  ( MultiArray1< T1, NumDims > &a,
    const MultiArray2< T2, NumDims, Allocator2 > &b, F f ) {
    typename MultiArray2< T2, NumDims, Allocator2 >::const_iterator j = b.begin();
    for ( typename MultiArray1< T1, NumDims >::iterator i = a.begin();
          i != a.end(); i++, j++ )
      multi_apply( *i, *j, f );
    return a;
  }

#if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 2))
  ///
  template<
    typename T1, typename T2, size_t NumDims, class F,
    typename Allocator1, typename Allocator2,
    template< typename, size_t, typename > class MultiArray1,
    template< typename, size_t, typename > class MultiArray2
  >
  MultiArray1< T1, NumDims, Allocator1 > &multi_apply
  ( MultiArray1< T1, NumDims, Allocator1 > &a,
    const MultiArray2< T2, NumDims, Allocator2 > &b, F f ) {
    typename MultiArray2< T2, NumDims, Allocator2 >::const_iterator j = b.begin();
    for ( typename MultiArray1< T1, NumDims, Allocator1 >::iterator i = a.begin();
          i != a.end(); i++, j++ )
      multi_apply( *i, *j, f );
    return a;
  }
#endif

  ///
  template<
    typename T1, typename T2, typename T3, class F,
    typename Allocator2, typename Allocator3,
    template< typename, size_t, typename > class MultiArray2,
    template< typename, size_t, typename > class MultiArray3
  >
  boost::detail::multi_array::sub_array< T1, 1 > multi_apply
  ( boost::detail::multi_array::sub_array< T1, 1 > a,
    const MultiArray2< T2, 1, Allocator2 > &b,
    const MultiArray3< T3, 1, Allocator3 > &c,
    F f ) {
    typename MultiArray2< T2, 1, Allocator2 >::const_iterator j = b.begin();
    typename MultiArray3< T3, 1, Allocator3 >::const_iterator k = c.begin();
    for ( typename boost::detail::multi_array::sub_array< T1, 1 >::iterator i = a.begin();
          i != a.end(); i++, j++, k++ )
      f( *i, *j, *k );
    return a;
  }

  ///
  template<
    typename T1, typename T2, typename T3, class F,
    typename Allocator2, typename Allocator3,
    template< typename, size_t > class MultiArray1,
    template< typename, size_t, typename > class MultiArray2,
    template< typename, size_t, typename > class MultiArray3
  >
  MultiArray1< T1, 1 > &multi_apply
  ( MultiArray1< T1, 1 > &a,
    const MultiArray2< T2, 1, Allocator2 > &b,
    const MultiArray3< T3, 1, Allocator3 > &c, F f ) {
    typename MultiArray2< T2, 1, Allocator2 >::const_iterator j = b.begin();
    typename MultiArray3< T3, 1, Allocator3 >::const_iterator k = c.begin();
    for ( typename MultiArray1< T1, 1 >::iterator i = a.begin();
          i != a.end(); i++, j++, k++ )
      f( *i, *j, *k );
    return a;
  }

#if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 2))
  ///
  template<
    typename T1, typename T2, typename T3, class F,
    typename Allocator1, typename Allocator2, typename Allocator3,
    template< typename, size_t, typename > class MultiArray1,
    template< typename, size_t, typename > class MultiArray2,
    template< typename, size_t, typename > class MultiArray3
  >
  MultiArray1< T1, 1, Allocator1 > &multi_apply
  ( MultiArray1< T1, 1, Allocator1 > &a,
    const MultiArray2< T2, 1, Allocator2 > &b,
    const MultiArray3< T3, 1, Allocator3 > &c, F f ) {
    typename MultiArray2< T2, 1, Allocator2 >::const_iterator j = b.begin();
    typename MultiArray3< T3, 1, Allocator3 >::const_iterator k = c.begin();
    for ( typename MultiArray1< T1, 1, Allocator1 >::iterator i = a.begin();
          i != a.end(); i++, j++, k++ )
      f( *i, *j, *k );
    return a;
  }
#endif

  ///
  template<
    typename T1, typename T2, typename T3, size_t NumDims, class F,
    typename Allocator2, typename Allocator3,
    template< typename, size_t, typename > class MultiArray2,
    template< typename, size_t, typename > class MultiArray3
  >
  boost::detail::multi_array::sub_array< T1, NumDims > multi_apply
  ( boost::detail::multi_array::sub_array< T1, NumDims > a,
    const MultiArray2< T2, NumDims, Allocator2 > &b,
    const MultiArray3< T3, NumDims, Allocator3 > &c,
    F f ) {
    typename MultiArray2< T2, NumDims, Allocator2 >::const_iterator j = b.begin();
    typename MultiArray3< T3, NumDims, Allocator3 >::const_iterator k = c.begin();
    for ( typename boost::detail::multi_array::sub_array< T1, NumDims >::iterator i = a.begin();
          i != a.end(); i++, j++, k++ )    multi_apply( *i, *j, *k, f );
    return a;
  }

  ///
  template<
    typename T1, typename T2, typename T3, size_t NumDims, class F,
    typename Allocator2, typename Allocator3,
    template< typename, size_t > class MultiArray1,
    template< typename, size_t, typename > class MultiArray2,
    template< typename, size_t, typename > class MultiArray3
  >
  MultiArray1< T1, NumDims > &multi_apply
  ( MultiArray1< T1, NumDims > &a,
    const MultiArray2< T2, NumDims, Allocator2 > &b,
    const MultiArray3< T3, NumDims, Allocator3 > &c, F f ) {
    typename MultiArray2< T2, NumDims, Allocator2 >::const_iterator j = b.begin();
    typename MultiArray3< T3, NumDims, Allocator3 >::const_iterator k = c.begin();
    for ( typename MultiArray1< T1, NumDims >::iterator i = a.begin();
          i != a.end(); i++, j++, k++ )
      multi_apply( *i, *j, *k, f );
    return a;
  }

#if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 2))
  ///
  template<
    typename T1, typename T2, typename T3, size_t NumDims, class F,
    typename Allocator1, typename Allocator2, typename Allocator3,
    template< typename, size_t, typename > class MultiArray1,
    template< typename, size_t, typename > class MultiArray2,
    template< typename, size_t, typename > class MultiArray3
  >
  MultiArray1< T1, NumDims, Allocator1 > &multi_apply
  ( MultiArray1< T1, NumDims, Allocator1 > &a,
    const MultiArray2< T2, NumDims, Allocator2 > &b,
    const MultiArray3< T3, NumDims, Allocator3 > &c, F f ) {
    typename MultiArray2< T2, NumDims, Allocator2 >::const_iterator j = b.begin();
    typename MultiArray3< T3, NumDims, Allocator3 >::const_iterator k = c.begin();
    for ( typename MultiArray1< T1, NumDims, Allocator1 >::iterator i = a.begin();
          i != a.end(); i++, j++, k++ )
      multi_apply( *i, *j, *k, f );
    return a;
  }
#endif

  ///
  template< typename T1, typename T2, class F >
  struct _multi_help1
  {
    _multi_help1( F _f ): f(_f) {}
    T1 &operator()( T1 &x, const T2 &y ) const
    { x = f( y ); return x; }
    F f;
  };

  ///
  template< typename T1, typename T2, typename T3, class F >
  struct _multi_help2
  {
    _multi_help2( F _f ): f(_f) {}
    T1 &operator()( T1 &x, const T2 &y, const T3 &z ) const
    { x = f( y, z ); return x; }
    F f;
  };

  ///
  template<
    typename T1, typename T2, size_t NumDims, class F, typename Allocator,
    template< typename, size_t, typename > class MultiArray
  >
  boost::multi_array< T1, NumDims > multi_func
  ( const MultiArray< T2, NumDims, Allocator > &a, F f ) {
    boost::multi_array< T1, NumDims > retVal( empty_clone< T1 >( a ) );
    return multi_apply( retVal, a, _multi_help1< T1, T2, F >( f ) );
  }

  ///
  template<
    typename T1, typename T2, class F
  >
  boost::detail::multi_array::sub_array< T1, 1 > multi_func
  ( const boost::detail::multi_array::sub_array< T2, 1 > a, F f ) {
    boost::multi_array< T1, 1 > retVal( empty_clone< T1 >( a ) );
    return multi_apply( retVal, a, _multi_help1< T1, T2, F >( f ) );
  }

  ///
  template<
    typename T1, typename T2, size_t NumDims, class F
  >
  boost::detail::multi_array::sub_array< T1, NumDims > multi_func
  ( const boost::detail::multi_array::sub_array< T2, NumDims > a, F f ) {
    boost::multi_array< T1, NumDims > retVal( empty_clone< T1 >( a ) );
    return multi_apply( retVal, a, _multi_help1< T1, T2, F >( f ) );
  }

  ///
  template<
    typename T1, typename T2, typename T3, size_t NumDims, class F,
    typename Allocator1, typename Allocator2,
    template< typename, size_t, typename > class MultiArray1,
    template< typename, size_t, typename > class MultiArray2
  >
  boost::multi_array< T1, NumDims > multi_func
  ( const MultiArray1< T2, NumDims, Allocator1 > &a,
    const MultiArray2< T3, NumDims, Allocator2 > &b, F f ) {
    boost::multi_array< T1, NumDims > retVal( empty_clone< T1 >( a ) );
    return multi_apply( retVal, a, b, _multi_help2< T1, T2, T3, F >( f ) );
  }

  ///
  template<
    typename T1, typename T2, size_t NumDims,
    template< typename, size_t > class MultiArray
  >
  boost::multi_array< T1, NumDims > multi_cast
  ( const MultiArray< T2, NumDims > &a ) {
    return multi_func< T1 >( a, std::_Identity< T2 >() );
  }

#if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 2))
  ///
  template<
    typename T1, typename T2, size_t NumDims,
    typename Allocator,
    template< typename, size_t, typename > class MultiArray
  >
  boost::multi_array< T1, NumDims > multi_cast
  ( const MultiArray< T2, NumDims, Allocator > &a ) {
    return multi_func< T1 >( a, std::_Identity< T2 >() );
  }

#endif

  ///@}

}

#define __MIMASINTERNALARRAYFUNC operator*=
#define __MIMASEXTERNALARRAYFUNC operator*
#define __MIMASFUNCTIONOBJECT std::multiplies
#include "multi_array_op_help.h"

#define __MIMASINTERNALARRAYFUNC operator/=
#define __MIMASEXTERNALARRAYFUNC operator/
#define __MIMASFUNCTIONOBJECT std::divides
#include "multi_array_op_help.h"

#define __MIMASINTERNALARRAYFUNC operator+=
#define __MIMASEXTERNALARRAYFUNC operator+
#define __MIMASFUNCTIONOBJECT std::plus
#include "multi_array_op_help.h"

#define __MIMASINTERNALARRAYFUNC operator-=
#define __MIMASEXTERNALARRAYFUNC operator-
#define __MIMASFUNCTIONOBJECT std::minus
#include "multi_array_op_help.h"

#define __MIMASEXTERNALARRAYFUNC absolute
#define __MIMASINTERNALARRAYFUNC absoluteIt
#define __MIMASFUNCTIONOBJECT _abs
#include "multi_array_op_help2.h"

#define __MIMASEXTERNALARRAYFUNC conj
#define __MIMASINTERNALARRAYFUNC conjIt
#define __MIMASFUNCTIONOBJECT _conj
#include "multi_array_op_help2.h"

#define __MIMASEXTERNALARRAYFUNC sqr
#define __MIMASINTERNALARRAYFUNC sqrIt
#define __MIMASFUNCTIONOBJECT _sqr
#include "multi_array_op_help2.h"

#define __MIMASEXTERNALARRAYFUNC logarithm
#define __MIMASINTERNALARRAYFUNC logarithmIt
#define __MIMASFUNCTIONOBJECT _log
#include "multi_array_op_help2.h"

#define __MIMASEXTERNALARRAYFUNC squareRoot
#define __MIMASINTERNALARRAYFUNC squareRootIt
#define __MIMASFUNCTIONOBJECT _sqrt
#include "multi_array_op_help2.h"

#define __MIMASEXTERNALARRAYFUNC sumSquares
#define __MIMASFUNCTIONOBJECT _sumsquares
#include "multi_array_op_help3.h"

#define __MIMASEXTERNALARRAYFUNC orientation
#define __MIMASFUNCTIONOBJECT _orientation
#include "multi_array_op_help3.h"

namespace mimas {
  /** @addtogroup arrayOp
      @{ */
  ///
  template <
    typename T1, typename T2, size_t NumDims,
    template< typename, size_t > class MultiArray
    >
  boost::multi_array< T1, NumDims > norm( const MultiArray< T2, NumDims > &a )
  {
    return multi_func< T1 >( a, _norm< T1, T2 >() );
  }

#if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 2))
  ///
  template <
    typename T1, typename T2, size_t NumDims,
    typename Allocator,
    template< typename, size_t, typename > class MultiArray
    >
    boost::multi_array< T1, NumDims > norm( const MultiArray< T2, NumDims, Allocator > &a )
  {
    return multi_func< T1 >( a, _norm< T1, T2 >() );
  }
#endif

  ///
  template <
    typename T1, typename T2, size_t NumDims,
    template< typename, size_t > class MultiArray
    >
  boost::multi_array< T1, NumDims > arg( const MultiArray< T2, NumDims > &a )
  {
    return multi_func< T1 >( a, _arg< T1, T2 >() );
  }

  ///
  template <
    typename T, size_t NumDims,
    template< typename, size_t > class MultiArray
    >
  boost::multi_array< int, NumDims > fastSqr( const MultiArray< T, NumDims > &a )
  {
    return multi_func< int >( a, _fastsqr< T >() );
  }

#if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 2))
  ///
  template <
    typename T, size_t NumDims, typename Allocator,
    template< typename, size_t, typename > class MultiArray
    >
  boost::multi_array< int, NumDims > fastSqr( const MultiArray< T, NumDims, Allocator > &a )
  {
    return multi_func< int >( a, _fastsqr< T >() );
  }
#endif

  ///@}
}

#endif
