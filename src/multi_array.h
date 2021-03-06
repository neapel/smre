#ifndef __MULTI_ARRAY_H__
#define __MULTI_ARRAY_H__

#include "multi_array_operators.h"
#include "multi_array_iterator.h"
#include "multi_array_fft.h"
#include "multi_array_print.h"

#include <boost/multi_array.hpp>
#include <array>
#include <algorithm>

namespace boost {
	typedef multi_array_types::index_range irange;

	template<class A>
	std::array<size_t, A::dimensionality> extents_of(const A &a) {
		std::array<size_t, A::dimensionality> out;
		std::copy(a.shape(), a.shape() + A::dimensionality, out.begin());
		return out;
	}
};

typedef std::array<size_t, 2> size2_t;


#endif
