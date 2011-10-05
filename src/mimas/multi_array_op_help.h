#if !defined(__MIMASINTERNALARRAYFUNC)
#   error "Do not include this file directly."
#endif
#if !defined(__MIMASEXTERNALARRAYFUNC)
#   error "Do not include this file directly."
#endif
#if !defined(__MIMASFUNCTIONOBJECT)
#   error "Do not include this file directly."
#endif

namespace mimas {

/** @addtogroup arrayOp
    @{ */
///
template <
  template< typename, size_t > class MultiArray1,
  template< typename, size_t > class MultiArray2,
  typename T, size_t NumDims
>
boost::multi_array< T, NumDims > __MIMASEXTERNALARRAYFUNC
( const MultiArray1< T, NumDims > &a,
  const MultiArray2< T, NumDims > &b )
{
  return multi_func< T >( a, b, __MIMASFUNCTIONOBJECT< T >() );
};

///
template <
  template< typename, size_t > class MultiArray,
  typename T, size_t NumDims
>
boost::multi_array< T, NumDims > __MIMASEXTERNALARRAYFUNC
( const MultiArray< T, NumDims > &a,
  const T &b )
{
  return multi_func< T >( a, std::bind2nd( __MIMASFUNCTIONOBJECT< T >(), b ) );
};

#if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 2))
///
template <
  template< typename, size_t, typename > class MultiArray,
  typename T, size_t NumDims, typename Allocator
>
boost::multi_array< T, NumDims > __MIMASEXTERNALARRAYFUNC
 ( const MultiArray< T, NumDims, Allocator > &a,
  const T &b )
{
  return multi_func< T >( a, std::bind2nd( __MIMASFUNCTIONOBJECT< T >(), b ) );
};
#endif

///
template <
  template< typename, size_t > class MultiArray,
  typename T, size_t NumDims
>
boost::multi_array< T, NumDims > __MIMASEXTERNALARRAYFUNC
( const T &a,
  const MultiArray< T, NumDims > &b )
{
  return multi_func< T >( b, std::bind1st( __MIMASFUNCTIONOBJECT< T >(), a ) );
};

#if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 2))
///
template <
  template< typename, size_t, typename > class MultiArray,
  typename T, size_t NumDims, typename Allocator
>
boost::multi_array< T, NumDims > __MIMASEXTERNALARRAYFUNC
( const T &a,
  const MultiArray< T, NumDims, Allocator > &b )
{
  return multi_func< T >( b, std::bind1st( __MIMASFUNCTIONOBJECT< T >(), a ) );
};
#endif

///
template <
  template< typename, size_t > class MultiArray2,
  typename T, size_t NumDims
>
boost::detail::multi_array::sub_array< T, NumDims > __MIMASINTERNALARRAYFUNC
( boost::detail::multi_array::sub_array< T, NumDims > a,
  const MultiArray2< T, NumDims > &b )
{
  return multi_apply( a, a, b,
                      _multi_help2< T, T, T, __MIMASFUNCTIONOBJECT< T > >
                        ( __MIMASFUNCTIONOBJECT< T >() ) );
};

///
template <
  template< typename, size_t > class MultiArray1,
  template< typename, size_t > class MultiArray2,
  typename T, size_t NumDims
>
MultiArray1< T, NumDims > &__MIMASINTERNALARRAYFUNC
( MultiArray1< T, NumDims > &a,
  const MultiArray2< T, NumDims > &b )
{
  return multi_apply( a, a, b,
                      _multi_help2< T, T, T, __MIMASFUNCTIONOBJECT< T > >
                      ( __MIMASFUNCTIONOBJECT< T >() ) );
};

///
template <
  typename T, size_t NumDims
>
boost::detail::multi_array::sub_array< T, NumDims > __MIMASINTERNALARRAYFUNC
( boost::detail::multi_array::sub_array< T, NumDims > a,
  const T &b )
{
  return multi_apply( a, a,
                      _multi_help1< T, T,
                          std::binder2nd< __MIMASFUNCTIONOBJECT< T > > >
                      ( std::bind2nd( __MIMASFUNCTIONOBJECT< T >(), b ) ) );
};

///
template <
  template< typename, size_t > class MultiArray,
  typename T, size_t NumDims
>
MultiArray< T, NumDims > &__MIMASINTERNALARRAYFUNC
( MultiArray< T, NumDims > &a,
  const T &b )
{
  return multi_apply( a, a,
                      _multi_help1< T, T,
                          std::binder2nd< __MIMASFUNCTIONOBJECT< T > > >
                      ( std::bind2nd( __MIMASFUNCTIONOBJECT< T >(), b ) ) );
};

#if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 2))
template <
  template< typename, size_t, typename > class MultiArray,
  typename T, size_t NumDims, typename Allocator
>
MultiArray< T, NumDims, Allocator > &__MIMASINTERNALARRAYFUNC
( MultiArray< T, NumDims, Allocator > &a,
  const T &b )
{
  return multi_apply( a, a,
                      _multi_help1< T, T,
                          std::binder2nd< __MIMASFUNCTIONOBJECT< T > > >
                      ( std::bind2nd( __MIMASFUNCTIONOBJECT< T >(), b ) ) );
};
#endif

///@}

}

#undef __MIMASINTERNALARRAYFUNC
#undef __MIMASEXTERNALARRAYFUNC
#undef __MIMASFUNCTIONOBJECT
