#ifndef __DEBUG_TOOLS_H__
#define __DEBUG_TOOLS_H__

#include <iostream>
#include <array>
#include <string>
#include <sstream>
#include <stdexcept>
#include <tuple>


namespace std {

/*! Print an array as <code>{a0, ..., an}</code> */
template<class T, size_t N>
ostream &operator<<(ostream &o, const array<T, N> &a) {
	o << '{';
	for(auto i = a.begin() ; i != a.end() ; i++) {
		if(i != a.begin()) o << ", ";
		o << *i;
	}
	o << '}';
	return o;
}

/*! Print a tuple as <code>(t0, ..., tn)</code>
 *
 * (iteration end, 1-tuple)
 */
template<size_t i = 0, typename... Args>
typename enable_if<i == sizeof...(Args) - 1, ostream &>::type
operator<<(ostream &o, const tuple<Args...> &t) {
	if(i == 0) o << '(';
	o << get<i>(t) << ')';
	return o;
}

/*! Print a tuple as <code>(t0, ..., tn)</code>
 *
 * (iteration, n-tuple, n > 1)
 */
template<size_t i = 0, typename... Args>
typename enable_if<i < sizeof...(Args) - 1, ostream &>::type
operator<<(ostream &o, const tuple<Args...> &t) {
	if(i == 0) o << '(';
	o << get<i>(t) << ", ";
	return operator<< <i + 1, Args...>(o, t);
}


}


// Throw an exception if the return value is not 0.
#define expect0(f, msg) { \
	int r = (f); \
	if(r != 0) { \
		std::ostringstream s; \
		s << (msg) << " Error: " << r; \
		throw std::runtime_error(s.str()); \
	} \
}


#endif
