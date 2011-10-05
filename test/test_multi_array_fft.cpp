#include "multi_array_fft.h"
#include "multi_array_print.h"
#include <iostream>

int main(int, char**) {
	using namespace boost;
	using namespace std;

	typedef multi_array<double, 2> array_t;
	typedef typename array_t::index index_t;


	array_t A(extents[5][5]);
	array_t A_out(extents[5][5]);

	auto dct = plan_dct(A, A_out);
	auto idct = plan_idct(A_out, A);

	int i = 0;
	for(auto r = A.begin() ; r != A.end() ; r++)
		for(auto c = r->begin() ; c != r->end() ; c++, i++)
			*c = i;

	dct();

	cout << A << endl;
	cout << A_out << endl;

	idct();

	cout << A << endl;
	cout << A_out << endl;


	return 0;
}
