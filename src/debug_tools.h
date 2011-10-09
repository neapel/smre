#ifndef __DEBUG_TOOLS_H__
#define __DEBUG_TOOLS_H__

#include <iostream>
#include <array>
#include <string>
#include <sstream>
#include <stdexcept>


namespace std {

// Print an array as "{a0, ..., an}"
template<class T, size_t N>
ostream &operator<<(ostream &o, array<T, N> &a) {
	o << '{';
	for(auto i = a.begin() ; i != a.end() ; i++) {
		if(i != a.begin()) o << ", ";
		o << *i;
	}
	o << '}';
	return o;
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
