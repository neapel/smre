#ifndef __MULTI_ARRAY_IO_H__
#define __MULTI_ARRAY_IO_H__

#include <gtkmm.h>
#include <iostream>
#include <memory>
#include <boost/regex.hpp>
#include <stdexcept>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <cmath>
#include "config.h"
#include "multi_array_io.h"

#include "chambolle_pock.h"
#include "constraint_parser.h"

using namespace std;
using namespace Gtk;
using namespace Gdk;
using namespace Glib;
using namespace sigc;


// Copy pixbuf data [0:255] into a new multi_array [-1:1]
boost::multi_array<float, 2> pixbuf_to_multi_array(const RefPtr<Pixbuf> &pb) {
	assert(pb->get_colorspace() == COLORSPACE_RGB);
	assert(pb->get_bits_per_sample() == 8);
	auto w = pb->get_width(), h = pb->get_height();
	auto px = pb->get_pixels();
	auto stride = pb->get_rowstride(), chans = pb->get_n_channels();
	boost::multi_array<float, 2> a(boost::extents[h][w]);
	for(int y = 0 ; y != h ; y++)
		for(int x = 0 ; x != w ; x++)
			a[y][x] = (px[y * stride + chans * x] / 255.0) * 2 - 1;
	return a;
}

// Copy multi_array data [-1:1] into a new pixbuf [0:255]
// mark_outliers: <-1 = red, >1 = blue, exactly 0 = green
RefPtr<Pixbuf> multi_array_to_pixbuf(const boost::multi_array<float, 2> &a) {
	auto h = a.shape()[0], w = a.shape()[1];
	auto pb = Pixbuf::create(COLORSPACE_RGB, false, 8, w, h);
	auto px = pb->get_pixels();
	auto stride = pb->get_rowstride();
	auto high = max(a), low = min(a);
	for(size_t y = 0 ; y != h ; y++)
		for(size_t x = 0 ; x != w ; x++) {
			auto p = px + (y * stride + 3 * x);
#if 1
			p[0] = p[1] = p[2] = high == low
				? 0
				: 255 * (a[y][x] - low) / (high - low);
			if(!isfinite(a[y][x])) {
				p[0] = 255; p[1] = p[2] = 0;
			}
#else
			auto f_val = a[y][x];
			if(mark_outliers && f_val == 0) { p[0] = p[2] = 0; p[1] = 200; }
			else {
				int value = 255 * (f_val + 1) / 2;
				if(!mark_outliers) value = max(0, min(255, value));
				if(value > 255) { p[0] = p[1] = 0; p[2] = 255; }
				else if(value < 0) { p[0] = 255; p[1] = p[2] = 0; }
				else { p[0] = p[1] = p[2] = value; }
			}
#endif
		}
	return pb;
}


#endif
