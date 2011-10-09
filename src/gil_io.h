#ifndef GIL_IO_H_
#define GIL_IO_H_

#include "smre.h"
#ifdef HAVE_GIL
#include "io.h"

// Read and write JPEG, TIFF & PNG image files using the Generic Image Library


class gil_reader : public file_reader {
	boost::multi_array<double, 2> data;
public:
	gil_reader(std::string filename);
	~gil_reader() { close(); }
	void close();
	operator bool();
	void operator>>(std::pair<std::string, data_t> &);
};

class gil_writer : public file_writer {
	boost::multi_array<double, 2> data;
public:
	gil_writer(std::string filename);
	~gil_writer() { close(); }
	void close();
	void operator<<(const std::pair<std::string, data_t> &);
};


class gil_io : public io_factory {
public:
	file_reader *reader(std::string) const;
	file_writer *writer(std::string) const;
};


#endif

#endif
