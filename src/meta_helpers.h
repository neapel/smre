#ifndef __META_HELPERS_H__
#define __META_HELPERS_H__

#include <cstddef>

namespace meta {

	
/*! A type to store list of integers */
template<std::size_t... ns>
struct integers {};

template<typename U, std::size_t n>
struct integers_push_back;

template<std::size_t n, std::size_t... ns>
struct integers_push_back<integers<ns...>, n> {
	typedef integers<ns..., n> type;
};


/*! This generates <code>integers<0, 1, 2, ..., n-1></code> */
template<std::size_t n>
struct iota {
	typedef typename integers_push_back<
		typename iota<n - 1>::type, n - 1
	>::type type;
};

template<>
struct iota<0> {
	typedef integers<> type;
};


};

#endif
