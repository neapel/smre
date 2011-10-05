#include <cassert>
#include "multi_array_operators.h"
#include "multi_array_print.h"

using namespace std;
using namespace boost;

int main(int, char**) {
	typedef multi_array<double, 4> array_t;
	typedef array_t::index index_t;

	array_t A( extents[2][2][4][3] );
	iota(A, 1);
	assert(A[0][1][2][1] == 20);
	A[0][1][2][1] = 12345;

	array<index_t, 4> bases {{-1, 1, -1, 4}};
	A.reindex(bases);

	cout << A << endl;

	// subarray
	cout << A[0] << endl;

	// view
	typedef multi_array_types::index_range range;
	array_t::index_gen gen;
	cout << A[ gen[-1][2][range(-1,3,2)][range(4,7)] ] << endl;
	

	return 0;
}

