#include "gil_io.h"
#ifdef HAVE_GIL

#include "multi_array_iterator.h"
#include <boost/algorithm/string.hpp>
#include <boost/mpl/vector.hpp>
//#include <boost/gil/gil_all.hpp>

#ifdef HAVE_JPEG
#	include <boost/gil/extension/io/jpeg_io.hpp>
#endif
#ifdef HAVE_PNG
#  include <boost/gil/extension/io/png_io.hpp>
#endif
#ifdef HAVE_TIFF
#	include <boost/gil/extension/io/tiff_io.hpp>
#endif


using namespace boost;
using namespace gil;
using namespace std;


template<class T>
static void read_any_format(string filename, T &img) {
#ifdef HAVE_JPEG
	try {
		img.recreate( jpeg_read_dimensions(filename) );
		jpeg_read_and_convert_image(filename, img);
		return;
	} catch(...) {}
#endif
#ifdef HAVE_PNG
	try {
		img.recreate( png_read_dimensions(filename) );
		png_read_and_convert_image(filename, img);
		return;
	} catch(...) {}
#endif
#ifdef HAVE_TIFF
	try {
		img.recreate( tiff_read_dimensions(filename) );
		tiff_read_and_convert_image(filename, img);
		return;
	} catch(...) {}
#endif
	throw runtime_error("No loader found for format");
}



multi_array<float, 2> read_image(string filename) {
	gray32_image_t input_image;
	read_any_format(filename, input_image);
	auto proper = view(input_image);
	const auto w = proper.width(), h = proper.height();
	multi_array<float, 2> data(extents[h][w]);
	auto data_view = interleaved_view(w, h,
			reinterpret_cast<gray32f_pixel_t *>( data.origin() ),
			sizeof(float) * data.strides()[0] );
	copy_and_convert_pixels(proper, data_view);
	return data;
}


void write_image(string filename, const multi_array<float, 2> &data) {
	auto data_view = interleaved_view( data.shape()[1], data.shape()[0],
		reinterpret_cast<const gray32f_pixel_t *>( data.origin() ),
		sizeof(float) * data.strides()[0] );
	gray8_image_t out_image(data_view.dimensions());
	copy_and_convert_pixels(data_view, view(out_image));
	png_write_view(filename, view(out_image));
}


#endif
