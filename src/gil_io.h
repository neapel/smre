#ifndef GIL_IO_H_
#define GIL_IO_H_

#include "smre.h"

boost::multi_array<float, 2> read_image(std::string);
void write_image(std::string, const boost::multi_array<float, 2> &);

#ifndef HAVE_GIL
#include <stdexcept>

boost::multi_array<float, 2> read_image(std::string) {
	throw std::runtime_error("Compiled without GIL, can't read images.");
}
void write_image(std::string, const boost::multi_array<float, 2> &) {
	throw std::runtime_error("Compiled without GIL, can't write images.");
}
#endif

#endif
