#ifndef __IMAGE_VARIANCE_H__
#define __IMAGE_VARIANCE_H__

#include "multi_array_operators.h"

template<class T>
T median(boost::multi_array<T,2> &a) {
	if(a.num_elements() == 0) throw std::invalid_argument("empty array");
	if(a.num_elements() == 1) return *a.data();
	std::sort(a.data(), a.data() + a.num_elements());
	const T *x = a.data() + (a.num_elements() - 1) / 2;
	return (x[0] + x[1]) / 2;
}

template<class T>
T median_absolute_deviation(boost::multi_array<T,2> a) {
	using namespace mimas;
	const T med = median(a);
	a -= med;
	absoluteIt(a);
	const T var = median(a);
	const T sig = 1.4826 * var;
	return sig;
}

#endif
