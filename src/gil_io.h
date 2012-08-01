#ifndef GIL_IO_H_
#define GIL_IO_H_

#include "smre.h"
#ifdef HAVE_GIL

boost::multi_array<float, 2> read_image(std::string);
void write_image(std::string, const boost::multi_array<float, 2> &);


#endif
#endif
