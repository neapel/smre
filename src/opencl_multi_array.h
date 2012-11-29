#ifndef __OPENCL_MULTI_ARRAY_H__
#define __OPENCL_MULTI_ARRAY_H__

#include "opencl_helpers.h"
#include "multi_array_operators.h"

namespace cl {

/** Read data from the buffer into the multi-array. */
template<template<typename, size_t, typename> class M, typename T, size_t N, typename A>
static inline buffer_read operator>>(buffer &b, M<T,N,A> &a) {
	if(!is_continuous(a))
		throw std::invalid_argument("Only supports continuous arrays.");
	return std::move( buffer_read(b, a.data(), sizeof(T) * a.num_elements()) );
}


/** Read date from the multi-array into the buffer. */
template<template<typename, size_t, typename> class M, typename T, size_t N, typename A>
static inline buffer_write operator<<(buffer &b, const M<T,N,A> &a) {
	if(!is_continuous(a))
		throw std::invalid_argument("Only supports continuous arrays.");
	return std::move( buffer_write(b, a.data(), sizeof(T) * a.num_elements()) );
}



};


#endif
