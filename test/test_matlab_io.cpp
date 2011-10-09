#include <iostream>
#include "multi_array_print.h"
#include "matlab_io.h"
#include "multi_array_compare.h"
#include "multi_array_operators.h"

using namespace std;
using namespace boost;

/* matfiles/test.mat, via GNU Octave:
  ans = B
  A =

     1   2   3

  B =

     4   5   6

  C =

     17   24    1    8   15
     23    5    7   14   16
      4    6   13   20   22
     10   12   19   21    3
     11   18   25    2    9
*/

int main(int argc, char **argv) {
	
	if(argc != 2) {
		cerr << "usage: " << argv[0] << " file.mat" << endl;
		return -1;
	}

	// read file
	{
		matlab_reader r(argv[1]);
		pair<string, data_t> a;
		r >> a;
		assert(a.first == "A");
		multi_equal(a.second, {1, 2, 3});

		r >> a;
		assert(a.first == "B");
		multi_equal(a.second, {4, 5, 6});

		r >> a;
		assert(a.first == "C");
		multi_equal(a.second, {
			17,   24,    1,    8,   15,
			23,    5,    7,   14,  16,
			4,    6,   13,   20,   22,
			10,   12,   19,   21,    3,
			11,   18,   25,    2,    9});
	}


	// write file
	{
		matlab_writer w("out0.mat");
		data_t data(extents[1][1][4][4]);
		iota(data, 0);
		ostringstream in; in << data;
		w << make_pair("test", data);
		w.close();

		matlab_reader r("out0.mat");
		pair<string, data_t> a;
		r >> a;
		assert(!r);
		ostringstream out; out << a.second;
		assert(a.first == "test");
		assert(out.str() == in.str());
	}

	
	// map interface
	{
		auto m = read_all("out0.mat");
		write_all("out1.mat", m);
	}

	return 0;
}
