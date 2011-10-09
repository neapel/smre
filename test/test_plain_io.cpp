#include "io.h"
#include "multi_array_compare.h"
#include "multi_array_operators.h"
#include "multi_array_print.h"
using namespace std;
using namespace boost;

int main(int, char**) {
	{
		map<string, data_t> in;
		in["test"].resize(extents[1][1][3][3]);
		iota(in["test"], 1);
		write_all("test0.txt", in);
	}

	{
		auto out = read_all("test0.txt");
		auto k = out["test"];
		cout << k << endl;
		assert(multi_equal(k, {1, 2, 3, 4, 5, 6, 7, 8, 9}));
	}

	return 0;
}
