#ifndef PLAIN_IO_H_
#define PLAIN_IO_H_

#include "smre.h"
#include "io.h"
#include <fstream>

// Format is:
// name2 5 7 3 :
//  1 2 3 4 5
//  6 7 8 9 10
//  11 12 13 14 15
// ...
// name2 7 8 9 :
//  1 2 ...

class plain_reader : public file_reader {
	std::ifstream s;
public:
	plain_reader(std::string filename);
	~plain_reader() { close(); }
	void close();
	operator bool();
	void operator>>(std::pair<std::string, data_t> &);
};

class plain_writer : public file_writer {
	std::ofstream s;
public:
	plain_writer(std::string filename);
	~plain_writer() { close(); }
	void close();
	void operator<<(const std::pair<std::string, data_t> &);
};


class plain_io : public io_factory {
public:
	file_reader *reader(std::string) const;
	file_writer *writer(std::string) const;
};


#endif
