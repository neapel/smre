#include "plain_io.h"
#include "multi_array_iterator.h"
#include <boost/algorithm/string.hpp>
#include <deque>
#include <iomanip>

using namespace boost;
using namespace std;

static const size_t N = data_t::dimensionality;


// Reader
plain_reader::plain_reader(string filename) : s(filename) {
	s >> skipws >> dec;
}

void plain_reader::close() {
	s.close();
}

plain_reader::operator bool() {
	return s.peek() != EOF;
}

void plain_reader::operator>>(pair<string, data_t> &out) {
	// variable name
	s >> out.first;

	// dimensions
	deque<size_t> rank;
	while(true) {
		char c;
		s >> c;
		if(c == ':') break;
		else s.putback(c);

		size_t r;
		s >> r;
		if(r == 0)
			throw runtime_error("Malformed input file.");
		rank.push_back(r);
	}
	if(rank.size() > N)
		throw runtime_error("Too many dimensions in input file.");
	while(rank.size() < N)
		rank.push_front(1);

	out.second.resize(rank);

	// data
	std::copy_n(istream_iterator<double>(s), out.second.num_elements(),
		all_elements(out.second));
}


// Writer
plain_writer::plain_writer(string filename) : s(filename) {
	s << dec << setprecision(9);
}

void plain_writer::close() {
	s.close();
}

void plain_writer::operator<<(const pair<string, data_t> &in) {
	// variable name
	s << in.first;

	// dimensions
	auto shape_start = in.second.shape();
	const auto shape_end = in.second.shape() + N;
	while(*shape_start == 1 && shape_start != shape_end) shape_start++;
	while(shape_start != shape_end)
		s << ' ' << *shape_start++;
	s << ":\n";

	// data
	size_t row = 0;
	const size_t row_length = in.second.shape()[N - 1];
	cout << "row" << row_length << endl;
	for(auto e : all_elements(in.second)) {
		s << e;
		if(row != row_length - 1) s << ' ';
		else s << '\n';
		row = (row + 1) % row_length;
	}
	s << flush;
}


// Factory
// Factories
static bool is_plain(const string &filename) {
	return iends_with(filename, ".txt");
}

file_reader *plain_io::reader(string filename) const {
	if(is_plain(filename))
		return new plain_reader(filename);
	return nullptr;
}

file_writer *plain_io::writer(string filename) const {
	if(is_plain(filename))
		return new plain_writer(filename);
	return nullptr;
}

