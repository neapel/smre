#ifndef __TUPLE_CUT_H__
#define __TUPLE_CUT_H__

#include <tuple>
#include "meta_helpers.h"


/*! Cuts the first n elements from a tuple, returns a new tuple. */
namespace tuple_cut_impl {


/*! Prepends a type to the tuple */
template<typename, typename>
struct cons;

template<typename Head, typename... Tail>
struct cons<Head, std::tuple<Tail...>> {
	typedef std::tuple<Head, Tail...> type;
};


/*! Returns the first n types of the tuple */
template<std::size_t, typename>
struct front;

template<typename Head, typename... Tail>
struct front<0, std::tuple<Head, Tail...>> {
	typedef std::tuple<> type;
};

template<>
struct front<0, std::tuple<>> {
	typedef std::tuple<> type;
};

template<std::size_t n, typename Head, typename... Tail>
struct front<n, std::tuple<Head, Tail...>> {
	typedef typename cons<
		Head,
		typename front<n - 1, std::tuple<Tail...>>::type
	>::type type;
};



/*! Creates a new tuple by copying some elements from the original.
 * Second parameter is the meta-list of indices.
 * @param t
 *   The original tuple.
 * */
template<
	std::size_t... i,
	typename... T
>
typename front<sizeof...(i), std::tuple<T...>>::type
select(const std::tuple<T...> &t, const meta::integers<i...>&) {
	return typename front<sizeof...(i), std::tuple<T...>>::type(
		std::get<i>(t)...
	);
}


/*! Create a tuple with the first n elements from the given tuple.
 * @param n
 *   Number of elements to cut (0 to length of tuple)
 * @param t
 *   The original tuple.
 */
template<std::size_t n, typename... T>
typename front<n, std::tuple<T...>>::type
tuple_cut(const std::tuple<T...> &t) {
	return select(
		t,
		typename meta::iota<n>::type()
	);
}

};

using tuple_cut_impl::tuple_cut;

#endif
