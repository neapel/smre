#ifndef __IO_H__
#define __IO_H__

#include "smre.h"
#include <string>
#include <map>

/*! Reads a file exactly once. */
class file_reader {
public:
	/*! Open the file for reading */
	// file_reader(std::string filename);

	/*! Close the file */
	virtual void close() = 0;

	/*! Read the next chunk from the file */
	virtual void operator>>(std::pair<std::string, data_t> &) = 0;

	/*! Return true if another chunk is available */
	virtual operator bool() = 0;
};


/*! Writes a file. */
class file_writer {
public:
	/*! Open the file for writing */
	// file_writer(std::string filename);

	/*! Close the file */
	virtual void close() = 0;

	/*! Write a chunk to the file */
	virtual void operator<<(const std::pair<std::string, data_t> &) = 0;
};


/*! Reader/writer factory */
class io_factory {
public:
	/*! Return a reader for the format or \c nullptr. */
	virtual file_reader *reader(std::string) const = 0;
	/*! Return a writer for the format or \c nullptr. */
	virtual file_writer *writer(std::string) const = 0;
};


/*! Return a reader for the file by guessing or detecting the type */
file_reader *reader_for(std::string filename);

/*! Return a writer for the file by guessing the type */
file_writer *writer_for(std::string filename);

/*! Return all the file's content as a map. */
std::map<std::string, data_t> read_all(std::string filename);

/*! Write a file from the given map. */
void write_all(std::string filename, const std::map<std::string, data_t> &);






#endif
