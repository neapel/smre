#ifndef __KERNEL_GENERATOR_H__
#define __KERNEL_GENERATOR_H__

#include "multi_array_operators.h"
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>
#include <stdexcept>

// returns a kernel with the given maximum size.
typedef std::function<boost::multi_array<float,2>(size_t max_w, size_t max_h)> kernel_generator;

// returns a function to create a gaussian kernel.
kernel_generator gaussian_kernel(float sigma) {
	using namespace boost;
	return [=](size_t w, size_t h) {
		multi_array<float, 2> out(extents[h][w]);
		const float den = -1 / (2 * pow(sigma, 2));
		float x0 = w/2.0, y0 = h/2.0;
		for(size_t y = 0 ; y < h ; y++)
			for(size_t x = 0 ; x < w ; x++)
				out[y][x] = exp( den * (pow(x - x0, 2) + pow(y - y0, 2)) );
		return out;
	};
}

// returns a function to create a box kernel.
kernel_generator box_kernel(size_t size) {
	using namespace boost;
	return [=](size_t, size_t) {
		multi_array<float, 2> k(extents[size][size]);
		fill(k, 1.0f/size);
		return k;
	};
}

// parse a kernel expression: "gauss:<sigma>" or "box:<size>".
kernel_generator kernel_from_string(std::string expr) {
	using namespace boost;
	regex r("gauss:(?<gauss>\\d+\\.?\\d*)|box:(?<box>\\d+)");
	smatch m;
	if(regex_match(expr, m, r)) {
		if(m["gauss"].matched)
			return gaussian_kernel(lexical_cast<float>(m["gauss"]));
		else if(m["box"].matched)
			return box_kernel(lexical_cast<size_t>(m["box"]));
	}
	throw std::invalid_argument("invalid kernel expression.");
}

#endif
