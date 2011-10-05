#ifndef __MULTI_ARRAY_OPERATORS_H__
#define __MULTI_ARRAY_OPERATORS_H__

#include "mimas/multi_array_op.h"

// Fill an array with increasing values
template<class I, class A>
void iota(A &a, I x = I()) {
	mimas::multi_apply(a, [&x](typename A::element &e){ e = x++; });
}


#if 0
#include <array>
#include <stdexcept>
#include <boost/multi_array.hpp>
// based on MIMAS multi_array_op.h


// Apply a function to every element in the array
template<class T, template<class, size_t> class A, class F>
void multi_apply(A<T, 1> &a, F f) {
	for(auto &i : a) f(i);
}

template<class T, template<class, size_t> class A, class F>
void multi_apply(const A<T, 1> &a, F f) {
	for(auto i : a) f(i);
}

template<class T, class Alloc, template<class, size_t, class> class A, class F>
void multi_apply(A<T, 1, Alloc> &a, F f) {
	for(auto &i : a) f(i);
}

template<class T, class Alloc, template<class, size_t, class> class A, class F>
void multi_apply(const A<T, 1, Alloc> &a, F f) {
	for(auto i : a) f(i);
}

template<class T, size_t N, template<class, size_t> class A, class F>
void multi_apply(A<T, N> &a, F f) {
	for(auto i : a) multi_apply(i, f);
}

template<class T, size_t N, template<class, size_t> class A, class F>
void multi_apply(const A<T, N> &a, F f) {
	for(auto i : a) multi_apply(i, f);
}

template<class T, size_t N, class Alloc, template<class, size_t, class> class A, class F>
void multi_apply(A<T, N, Alloc> &a, F f) {
	for(auto i : a) multi_apply(i, f);
}

template<class T, size_t N, class Alloc, template<class, size_t, class> class A, class F>
void multi_apply(const A<T, N, Alloc> &a, F f) {
	for(auto i : a) multi_apply(i, f);
}


// Fill an array with increasing values
template<class I, class A>
void iota(A &a, I x = I()) {
	multi_apply(a, [&x](typename A::element &e){ e = x++; });
}


// Compute the 2-norm of this array
template<class A>
typename A::element square_sum(const A &a) {
	typename A::element sum;
	multi_apply(a, [&sum](const typename A::element &e){ sum += e * e; });
	return sum;
}



// Combine two arrays elementwise
template<
	class T1, template<class, size_t> class A1,
	class T2, template<class, size_t> class A2,
	class F,
	class T3, template<class, size_t> class A3
>
void multi_combine(const A1<T1, 1> &a, const A2<T2, 1> &b, F f, A3<T3, 1> &out) {
/*	if(a.size() != b.size() || b.size() != out.size())
		throw std::invalid_argument("Arrays of different sizes.");
	auto i = a.begin(), j = b.begin(), k = out.begin();
	for( ; i != a.end() ; i++, j++, k++)
		*k = f(*i, *j);*/
}

template<
	class T1, class Alloc1, template<class, size_t, class> class A1,
	class T2, class Alloc2, template<class, size_t, class> class A2,
	class F,
	class T3, class Alloc3, template<class, size_t, class> class A3
>
void multi_combine(const A1<T1, 1, Alloc1> &a, const A2<T2, 1, Alloc2> &b, F f, A3<T3, 1, Alloc3> &out) {
/*	if(a.size() != b.size() || b.size() != out.size())
		throw std::invalid_argument("Arrays of different sizes.");
	auto i = a.begin(), j = b.begin(), k = out.begin();
	for( ; i != a.end() ; i++, j++, k++)
		*k = f(*i, *j);*/
}

template<
	size_t N,
	class T1, template<class, size_t> class A1,
	class T2, template<class, size_t> class A2,
	class F,
	class T3, template<class, size_t> class A3
>
void multi_combine(const A1<T1, N> &a, const A2<T2, N> &b, F f, A3<T3, N> &out) {
/*	if(a.size() != b.size() || b.size() != out.size())
		throw std::invalid_argument("Arrays of different sizes.");
	auto i = a.begin(), j = b.begin(), k = out.begin();
	for( ; i != a.end() ; i++, j++, k++)
		multi_combine(*i, *j, f, *k);*/
}

template<
	size_t N,
	class T1, class Alloc1, template<class, size_t, class> class A1,
	class T2, class Alloc2, template<class, size_t, class> class A2,
	class F,
	class T3, class Alloc3, template<class, size_t, class> class A3
>
void multi_combine(const A1<T1, N, Alloc1> &a, const A2<T2, N, Alloc2> &b, F f, A3<T3, N, Alloc3> &out) {
	if(a.size() != b.size() || b.size() != out.size())
		throw std::invalid_argument("Arrays of different sizes.");
	auto i = a.begin();
	auto j = b.begin();
	auto k = out.begin();
	for( ; i != a.end() ; i++, j++, k++) {
		multi_combine(*i, *j, f, *k);
	}
}

template<class T, class A>
boost::multi_array<T, A::dimensionality> empty_clone(const A &a) {
	std::array<size_t, A::dimensionality> shape;
	std::copy(&a.shape()[0], &a.shape()[A::dimensionality], shape.begin());
	return boost::multi_array<T, A::dimensionality>(shape);
}


template<class A, class B, class F>
auto multi_combine(const A &a, const B &b, F f)
-> boost::multi_array<decltype(f(*a.origin(), *b.origin())), A::dimensionality> {
	static_assert(A::dimensionality == B::dimensionality, "Arrays must have same rank");
	auto out = empty_clone<decltype(f(*a.origin(), *b.origin()))>(a);
	multi_combine(a, b, f, out);
	return out;
}


// Elementwise scalar operators
template<class A, class T>
void operator*=(A &a, T factor) {
	multi_apply(a, [factor](typename A::element &x){ x *= factor; });
}

template<class A, class T>
void operator/=(A &a, T factor) {
	multi_apply(a, [factor](typename A::element &x){ x /= factor; });
}

template<class A, class T>
void operator+=(A &a, T factor) {
	multi_apply(a, [factor](typename A::element &x){ x += factor; });
}

template<class A, class T>
void operator-=(A &a, T factor) {
	multi_apply(a, [factor](typename A::element &x){ x -= factor; });
}


// Elementwise array operators
template<class A, class B>
auto operator+(const A &a, const B &b)
-> boost::multi_array<decltype(*a.origin() + *b.origin()), A::dimensionality> {
	return multi_combine(a, b, [](const typename A::element &x, const typename B::element &y){ return x + y; });
}

template<class A, class B>
auto operator-(const A &a, const B &b)
-> boost::multi_array<decltype(*a.origin() - *b.origin()), A::dimensionality> {
	return multi_combine(a, b, [](const typename A::element &x, const typename B::element &y){ return x - y; });
}


// Compare two arrays for approximate equality
template<class A, class B, class T>
bool equals(const A &a, const B &b, T epsilon) {
	return square_sum(a - b) < pow(epsilon, 2);
}

template<class A, class B>
bool equals(const A &a, const B &b) {
	return equals(a, b, 1e-6);
}
#endif

#endif
