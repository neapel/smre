#ifndef MATLAB_IO_H_
#define MATLAB_IO_H_

#include "smre.h"
#ifdef HAVE_MATIO

#include <matio.h>
#include "io.h"

class matlab_reader : public file_reader {
	mat_t *mat;
	matvar_t *var;
	void read();

public:
	/*! Open a MATLAB file */
	matlab_reader(std::string filename);

	~matlab_reader() { close(); }

	/*! Closes the file */
	void close();

	/*! Returns true if more entries can be read */
	operator bool();

	/*! Read one entry */
	void operator>>(std::pair<std::string, data_t> &);
};

class matlab_writer : public file_writer {
	mat_t *mat;

public:
	/*! Write a MATLAB file. Truncates existing files. */
	matlab_writer(std::string filename);

	~matlab_writer() { close(); }

	/*! Close the file */
	void close();

	/*! Write one entry */
	void operator<<(const std::pair<std::string, data_t> &);
};



/*! Reader/writer factory class.
 * Returns NULL if the file can't be read/written */
class matlab_io : public io_factory {
public:
	file_reader *reader(std::string) const;
	file_writer *writer(std::string) const;
};

#endif

#endif /* MATLAB_IO_H_ */
