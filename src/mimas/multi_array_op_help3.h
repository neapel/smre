#ifndef __MIMASEXTERNALARRAYFUNC
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
boost::multi_array< T, NumDims > __MIMASEXTERNALARRAYFUNC( const MultiArray< T, NumDims > &a, const MultiArray< T, NumDims > &b )
{
  return multi_func< T >( a, b, __MIMASFUNCTIONOBJECT< T >() );
};

///@}

}

#undef __MIMASEXTERNALARRAYFUNC
#undef __MIMASFUNCTIONOBJECT
