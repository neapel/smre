#ifndef __MULTI_ARRAY_OPERATORS_H__
#define __MULTI_ARRAY_OPERATORS_H__

#include "multi_array_iterator.h"
#include "mimas/multi_array_op.h"
#include <complex>

/** Fills an array with increasing values */
template<class I, class A>
void iota(A &a, I x = I()) {
	mimas::multi_apply(a, [&x](typename A::element &e){ e = x++; });
}


/** Fills an array with a value */
template<class A>
void fill(A &a, typename A::element x) {
	mimas::multi_apply(a, [x](typename A::element &e){ e = x; });
}


/** Returns the minimum value from the array */
template<class A>
typename A::element min(const A &a) {
	typename A::element x = a[0][0];
	mimas::multi_apply(const_cast<A&>(a), [&x](const typename A::element &e){ if(e < x) x = e; });
	return x;
}

/** Returns the maximum value from the array */
template<class A>
typename A::element max(const A &a) {
	typename A::element x = a[0][0];
	mimas::multi_apply(const_cast<A&>(a), [&x](const typename A::element &e){ if(e > x) x = e; });
	return x;
}

/** Returns the sum of all elements of the array */
template<class A>
typename A::element sum(const A &a) {
	typename A::element x = 0;
	mimas::multi_apply(const_cast<A&>(a), [&x](const typename A::element &e){ x += e; });
	return x;
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
void kernel_pad(const boost::multi_array<T1, 2> &in, boost::multi_array<T2, 2> &out, bool conjugate = false) {
	const size_t w = out.shape()[0], h = out.shape()[1], kw = in.shape()[0], kh = in.shape()[1];
	const size_t dx = conjugate ? (w - kw + 1) : 0, dy = conjugate ? (h - kh + 1) : 0;

	for(size_t x = 0 ; x < kw ; x++)
		for(size_t y = 0 ; y < kh ; y++)
			out[(x+dx) % w][(y+dy) % h] = in[x][y];
}


namespace std {
	inline double conj(const double &x) { return x; }
	inline float conj(const float &x) { return x; }
};


/** Return conjugated transposed array. */
template<class T>
boost::multi_array<T, 2> conjugate_transpose(const boost::multi_array<T, 2> &in) {
	size_t w = in.shape()[0], h = in.shape()[1];
	boost::multi_array<T, 2> out(boost::extents[w][h]);
	for(size_t x = 0 ; x < w ; x++)
		for(size_t y = 0 ; y < h ; y++)
			out[x][y] = std::conj(in[y][x]);
	return out;
}


#endif
