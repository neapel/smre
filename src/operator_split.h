#ifndef __OPERATOR_SPLIT_H__
#define __OPERATOR_SPLIT_H__

#include <cstddef>
#include <tuple>
#include <functional>


#include "apply_tuple.h"
#include "meta_helpers.h"
#include "tuple_cut.h"

/*! Operator Split internal implementation */
namespace operator_split_impl {

/*! Put a type to the front of the argument list */
template<typename T, typename U>
struct cons;

template<typename Return, typename... Tail, typename Head>
struct cons<Head, Return(Tail...)> {
	typedef Return type(Head, Tail...);
};


/*! This converts <code>std::tuple<T0, T1, ..., TN></code> to the function type
 * <code>TK(T0, T1, ..., TK-1)</code> where K is the first parameter */
template<typename, std::size_t, typename...>
struct one_function;

template<typename All, std::size_t k, typename Head, typename... Tail>
struct one_function<All, k, Head, Tail...> {
	typedef typename cons<
		Head,
		typename one_function<All, k - 1, Tail..., All>::type
	>::type type;
};

template<typename Head, typename... Tail, typename... All>
struct one_function<std::tuple<All...>, 0, Head, Tail...> {
    typedef Head type(All...);
};


/*! Creates all functions for the tuple and a list of numbers. */
template<typename T, typename U>
struct functions_helper;

template<typename... T, std::size_t... n>
struct functions_helper<std::tuple<T...>, meta::integers<n...>> {
	typedef std::tuple<
		std::function<
			typename one_function<
				std::tuple<T...>,
				n,
				T...
			>::type
		>...
	> type;
};


/*! Creates all functions for the tuple. */
template<typename... T>
struct functions {
	typedef typename functions_helper<
		std::tuple<T...>,
		typename meta::iota<sizeof...(T)>::type
	>::type type;
};


/*! End of iteration, does nothing. */
template<std::size_t i = 0, typename... T>
typename std::enable_if<i == sizeof...(T), void>::type
iterate(std::tuple<T...>&, const std::tuple<T...>&, const typename functions<T...>::type &) {}


/*! Sets the i'th value of tuple <code>values</code> to the output of the i'th function.
 * @param values The new tuple to build.
 * @param old_values The old values to use.
 * @param functions The functions to use.
 */
template<std::size_t i = 0, typename... T>
typename std::enable_if<i < sizeof...(T), void>::type
iterate(std::tuple<T...> &values, const std::tuple<T...> &old_values, const typename functions<T...>::type &functions) {
	// store the output of the function call
	std::get<i>(values) = apply_tuple(
		std::get<i>(functions),
		// arguments include: new values calculated by previous functions
		// all old values from previous iteration.
		std::tuple_cat(
			tuple_cut<i>(values),
			old_values
		)
	);
	// iterate further.
	iterate<i + 1>(values, old_values, functions);
}


/*!
 * Operator Splitting.
 *
 * Works like this, except for an arbitrary number of types:
 *
 * <pre>
 * tuple<T0, T1> operator_split(
 *   tuple<T0, T1> values,
 *   tuple<function<T0(T0,T1)>, function<T1(T0,T0,T1)>> functions,
 *   function<bool(T0,T1)> if_continue
 * ) {
 *   while(if_continue(values)) {
 *     old_values = values;
 *     values[0] = functions[0](old_values[0], old_values[1]);
 *     values[1] = functions[1](values[0], old_values[0], old_values[1]);
 *   }
 *   return values;
 * }
 * </pre>
 *
 * @param values
 *   A tuple of initial values, T0 × T1 × … × TN
 * @param functions
 *   A tuple of functions which calculate the n'th new value from the current values computed by the functions before, and previous values: tuple<(T0 × T1 × …) → T0, (T0 × T0 × T1 …) → T1, … → T2, …>
 * @param if_continue
 *   Continue as long as this function returns <code>true</code>
 */
template<typename... T>
std::tuple<T...> 
operator_split(
	std::tuple<T...> values,
	typename functions<T...>::type functions,
	std::function<bool(T...)> if_continue
) {
	while(apply_tuple(if_continue, values)) {
		// *copy* old values
		std::tuple<T...> old_values = values;
		// calculate new values
		iterate(values, old_values, functions);
	}
	return values;
}

};


using operator_split_impl::operator_split;


#endif
