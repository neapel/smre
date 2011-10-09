#include "gil_io.h"
#ifdef HAVE_GIL

#include "multi_array_iterator.h"
#include <boost/algorithm/string.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/gil/gil_all.hpp>

#ifdef HAVE_JPEG
#	include <boost/gil/extension/io/jpeg_dynamic_io.hpp>
#endif
#ifdef HAVE_PNG
#endif
#ifdef HAVE_TIFF
#endif


using namespace boost;
using namespace gil;
using namespace std;


static const size_t N = data_t::dimensionality;


// Reader
gil_reader::gil_reader(string filename) {
	typedef mpl::vector<gray8_image_t> my_img_types;
	any_image<my_img_types> runtime_image;
	jpeg_read_image(filename, runtime_image);
}

void gil_reader::close() {
}

gil_reader::operator bool() {
}

void gil_reader::operator>>(pair<string, data_t> &out) {
}


// Writer
gil_writer::gil_writer(string filename) {
}

void gil_writer::close() {
}

void gil_writer::operator<<(const pair<string, data_t> &in) {
}


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

file_writer *gil_io::writer(string filename) const {
	if(is_gil(filename))
		return new gil_writer(filename);
	return nullptr;
}


#endif
