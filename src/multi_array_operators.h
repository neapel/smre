#ifndef __MULTI_ARRAY_OPERATORS_H__
#define __MULTI_ARRAY_OPERATORS_H__

#include "mimas/multi_array_op.h"
#include <complex>

/** Fills an array with increasing values */
template<class I, class A>
void iota(A &a, I x = I()) {
	mimas::multi_apply(a, [&x](typename A::element &e){ e = x++; });
}


/** Returns true if there are no unused bytes in the array's memory */
template<class A>
bool is_continuous(const A &a) {
	typename A::size_type size = a.shape()[0];
	for(size_t i = 0 ; i < a.num_dimensions() ; i++)
		size *= a.strides()[i];
	return size == a.num_elements();
}


/** Copies the elements of a kernel into the output array such that
 * the kernel center(=origin) is at the corners of the output array. 
 * Doesn't modify cells not covered by the kernel, use mimas::operator= */
template<class T1, class T2>
void kernel_pad(const boost::multi_array<T1, 2> &in, boost::multi_array<T2, 2> &out) {
	const size_t w = out.shape()[0], h = out.shape()[1], kw = in.shape()[0], kh = in.shape()[1];
	// kernel origin = corner
	const size_t dx = w - kw/2;
	const size_t dy = h - kh/2;
	for(size_t y = 0 ; y < kh ; y++)
		for(size_t x = 0 ; x < kw ; x++)
			out[(y+dy) % h][(x+dx) % w] = in[y][x];
}


namespace std {
	inline double conj(const double &x) { return x; }
	inline float conj(const float &x) { return x; }
};


/** Return conjugated transposed array. */
template<class T>
boost::multi_array<T, 2> conjugate_transpose(const boost::multi_array<T, 2> &in) {
	size_t w = a.shape()[0], h = a.shape()[1];
	boost::multi_array<T, 2> out(extents[w][h]);
	for(size_t x = 0 ; x < w ; x++)
		for(size_t y = 0 ; y < h ; y++)
			out[x][y] = std::conj(in[y][x]);
}


#endif
