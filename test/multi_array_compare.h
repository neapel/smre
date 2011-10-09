#ifndef __MULTI_ARRAY_COMPARE_H__
#define __MULTI_ARRAY_COMPARE_H__

#include "multi_array_iterator.h"
#include <iostream>


// compare multi_array view to the parameters
template<class A>
bool multi_equal(const A &a, std::initializer_list<typename A::element> e) {
	const_element_iterator<A> begin(a), end = begin.end(); 
	if(end - begin != e.end() - e.begin()) return false;
	return std::equal(begin, end, e.begin());
}


// compare a Container to the parameters
template<class V>
bool range_equal(const V &v, std::initializer_list<typename V::value_type> e) {
	if(v.size() != e.size()) return false;
	return std::equal(v.begin(), v.end(), e.begin());
}


// compare a native array
template<class T, class T2>
bool c_equal(const T *a, std::initializer_list<T2> e) {
	return std::equal(a, a + e.size(), e.begin());
}


// print a native array
template<size_t N, class A>
std::ostream &print_array(const A a) {
	for(size_t i = 0 ; i != N ; i++)
		std::cout << a[i] << ' ';
	return std::cout;
}

// print a new array
#include <iterator>
template<class T, size_t N>
std::ostream &operator<<(std::ostream &o, const std::array<T, N> &a) {
	o << '{';
	std::copy(a.begin(), a.end(), std::ostream_iterator<T>(o, ", "));
	o << '}';
	return o;
}


#endif
