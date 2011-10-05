#ifndef __MULTI_ARRAY_COMPARE_H__
#define __MULTI_ARRAY_COMPARE_H__

#include "multi_array_iterator.h"
#include <cstdarg>


// compare multi_array view to the parameters
template<class A>
bool multi_equal(const A &a, std::initializer_list<typename A::element> e) {
	const_element_iterator<A> begin(a), end = begin.end(); 
	if(end - begin != e.end() - e.begin()) return false;
	return std::equal(begin, end, e.begin());
}


// compare ublas vector to the parameters
template<class V>
bool blas_equal(const V &v, std::initializer_list<typename V::value_type> e) {
	if(v.size() != e.size()) return false;
	return std::equal(v.begin(), v.end(), e.begin());
}


#endif
