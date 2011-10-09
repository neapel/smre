#include "multi_array_operators.h"
#include "multi_array_print.h"
#include <iostream>
#include <typeinfo>
using namespace boost;
using namespace std;
using namespace mimas;



int main(int, char**) {
	multi_array<double, 2> A(extents[4][3]);
	iota(A, 1);
	double v = 8;
	assert(A[2][1] == v);

	A *= 4.0;
	v *= 4.0; assert(A[2][1] == v);

	A /= 3.0;
	v /= 3.0; assert(A[2][1] == v);

	A += 4.0;
	v += 4.0; assert(A[2][1] == v);

	A -= 1.0;
	v -= 1.0; assert(A[2][1] == v);

	return 0;
}
