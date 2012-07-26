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


static const size_t N = data_t::dimensionality;

static const std::array<string, 3> channel_names{{ "r", "g", "b" }};

template<class Image>
static void read_image(string filename, Image &img) {
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


// Reader
gil_reader::gil_reader(string filename) : channel(0) {
	rgb16_image_t img;
	read_image(filename, img);

	auto proper = view(img);

	data.resize(extents[proper.num_channels()][proper.height()][proper.width()]);

	for(size_t i = 0 ; i < proper.num_channels() ; i++) {
		const auto chan = nth_channel_view(proper, i);
		auto plane = data[i];
		copy(chan.begin(), chan.end(), all_elements(plane));
	}
}

void gil_reader::close() {
}

gil_reader::operator bool() {
	return channel < data.shape()[0];
}

void gil_reader::operator>>(pair<string, data_t> &out) {
	out.first = channel_names[channel];

	std::array<size_t, N> size;
	fill(size.begin(), size.end(), 1);
	copy_backward(data.shape() + 1, data.shape() + 3, size.end());
	out.second.resize(size);
	const auto in_data = all_elements(data[channel]);
	copy(in_data.begin(), in_data.end(), all_elements(out.second));

	channel++;
}


// Writer
#if 0
gil_writer::gil_writer(string filename) : filename(filename), data() {
}

void gil_writer::close() {
}

void gil_writer::operator<<(const pair<string, data_t> &in) {

	if(data.num_elements() == 0)
		data.resize(extents[1][in.second.shape()[N - 2]][in.second.shape()[N - 1]]);

}
#endif

// Factory
static bool is_gil(const string &filename) {
	return false
#ifdef HAVE_JPEG
		|| iends_with(filename, ".jpg")
		|| iends_with(filename, ".jpeg")
#endif
#ifdef HAVE_PNG
		|| iends_with(filename, ".png")
#endif
#ifdef HAVE_TIFF
		|| iends_with(filename, ".tif")
		|| iends_with(filename, ".tiff")
#endif
	;
}

file_reader *gil_io::reader(string filename) const {
	if(is_gil(filename))
		return new gil_reader(filename);
	return nullptr;
}

file_writer *gil_io::writer(string) const {
#if 0
	if(is_gil(filename))
		return new gil_writer(filename);
#endif
	return nullptr;
}


#endif
