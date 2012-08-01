#include "io.h"

// loaders
#include "matlab_io.h"
#include "plain_io.h"

#include <stdexcept>

using namespace std;

static io_factory *factories[] = {
#ifdef HAVE_MATIO
	new matlab_io(),
#endif
	new plain_io()
};


file_reader *reader_for(string filename) {
	for(const io_factory *f : factories) {
		file_reader *r = f->reader(filename);
		if(r) return r;
	}
	throw invalid_argument("No reader for this format.");
}

file_writer *writer_for(string filename) {
	for(const io_factory *f : factories) {
		file_writer *w = f->writer(filename);
		if(w) return w;
	}
	throw invalid_argument("No writer for this format.");
}


map<string, data_t> read_all(string filename) {
	file_reader *r = reader_for(filename);
	map<string, data_t> all;

	while(*r) {
		pair<string, data_t> elem;
		*r >> elem;
		all.insert(elem);
	}
	delete r;
	return all;
}

void write_all(string filename, const map<string, data_t> &all) {
	file_writer *w = writer_for(filename);
	for(const auto elem : all)
		*w << elem;
	delete w;
}


