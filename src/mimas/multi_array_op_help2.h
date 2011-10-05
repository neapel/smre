#ifndef __MIMASEXTERNALARRAYFUNC
#   error "Do not include this file directly."
#endif
#ifndef __MIMASINTERNALARRAYFUNC
#   error "Do not include this file directly."
#endif
#ifndef __MIMASFUNCTIONOBJECT
#   error "Do not include this file directly."
#endif

namespace mimas {

/** @addtogroup arrayOp
    @{ */
///
template <
  typename T, size_t NumDims,
  template< typename, size_t > class MultiArray
>
MultiArray< T, NumDims > &__MIMASINTERNALARRAYFUNC( MultiArray< T, NumDims > &a )
{
  return multi_apply( a, a, _multi_help1< T, T, __MIMASFUNCTIONOBJECT< T > >
                      ( __MIMASFUNCTIONOBJECT< T >() ) );
};

#if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 2))
///
template <
  typename T, size_t NumDims, typename Allocator,
  template< typename, size_t, typename > class MultiArray
>
MultiArray< T, NumDims, Allocator > &__MIMASINTERNALARRAYFUNC( MultiArray< T, NumDims, Allocator > &a )
{
  return multi_apply( a, a, _multi_help1< T, T, __MIMASFUNCTIONOBJECT< T > >
                      ( __MIMASFUNCTIONOBJECT< T >() ) );
};
#endif

///
template <
  typename T, size_t NumDims,
  template< typename, size_t > class MultiArray
>
boost::multi_array< T, NumDims > __MIMASEXTERNALARRAYFUNC( const MultiArray< T, NumDims > &a )
{
  return multi_func< T >( a, __MIMASFUNCTIONOBJECT< T >() );
};

#if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 2))
///
template <
  typename T, size_t NumDims, typename Allocator,
  template< typename, size_t, typename > class MultiArray
>
boost::multi_array< T, NumDims > __MIMASEXTERNALARRAYFUNC( const MultiArray< T, NumDims, Allocator > &a )
{
  return multi_func< T >( a, __MIMASFUNCTIONOBJECT< T >() );
};
#endif

///@}
}

#undef __MIMASEXTERNALARRAYFUNC
#undef __MIMASINTERNALARRAYFUNC
#undef __MIMASFUNCTIONOBJECT
