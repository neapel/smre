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

	cout << A << endl;

	A *= 4.0;
	cout << A << endl;

	A /= 3.0;
	cout << A << endl;

	A += 4.0;
	cout << A << endl;

	A -= 1.0;
	cout << A << endl;


	return 0;
}
