#include <cassert>
#include "multi_array.h"
#include "multi_array_print.h"

using namespace std;
using namespace boost;

int main(int, char**) {
	multi_array<double, 4> A( extents[2][2][4][3] );
	iota(A, 1);
	assert(A[0][1][2][1] == 20);
	A[0][1][2][1] = 12345;

	A.reindex(std::array<int, 4>{{-1, 1, -1, 4}});

	cout << A << endl;

	// subarray
	cout << A[0] << endl;

	// view
	cout << A[ indices[-1][2][irange(-1,3,2)][irange(4,7)] ] << endl;
	
	// exact string
	{
		multi_array<int, 2> B(extents[3][3]);
		iota(B, 1);
		ostringstream s;
		s << B;
		assert(s.str() ==
			"⎛1 2 3⎞\n"
			"⎜4 5 6⎟\n"
			"⎝7 8 9⎠"
		);
	}

	return 0;
}

