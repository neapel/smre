#ifndef __MULTI_ARRAY_IO_H__
#define __MULTI_ARRAY_IO_H__

#include <gtkmm.h>
#include <boost/multi_array.hpp>

// Copy pixbuf data [0:255] into a new multi_array [-1:1]
boost::multi_array<float, 2> pixbuf_to_multi_array(const Glib::RefPtr<Gdk::Pixbuf> &pb) {
	assert(pb->get_colorspace() == Gdk::COLORSPACE_RGB);
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
Glib::RefPtr<Gdk::Pixbuf> multi_array_to_pixbuf(const boost::multi_array<float, 2> &a, bool auto_scale = false) {
	auto h = a.shape()[0], w = a.shape()[1];
	auto pb = Gdk::Pixbuf::create(Gdk::COLORSPACE_RGB, false, 8, w, h);
	auto px = pb->get_pixels();
	auto stride = pb->get_rowstride();
	float min = 0, max = 0;
	if(auto_scale) {
		for(size_t y = 0 ; y != h ; y++)
			for(size_t x = 0 ; x != w ; x++) {
				if(a[y][x] > max) max = a[y][x];
				if(a[y][x] < min) min = a[y][x];
			}
	} else {
		min = -1;
		max = 1;
	}
	for(size_t y = 0 ; y != h ; y++)
		for(size_t x = 0 ; x != w ; x++) {
			auto p = px + (y * stride + 3 * x);
			float v = (a[y][x] - min) / (max - min) * 255;
			if(v < 0) v = 0;
			if(v > 255) v = 255;
			p[0] = p[1] = p[2] = v;
		}
	return pb;
}


#endif
