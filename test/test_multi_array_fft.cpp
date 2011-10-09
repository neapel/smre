#include "multi_array_fft.h"
#include "multi_array_print.h"
#include "multi_array_operators.h"
#include "multi_array_iterator.h"
#include "multi_array_compare.h"
#include <iostream>

using namespace boost;
using namespace std;

int main(int, char**) {
	typedef multi_array_types::index_range range;

	// normal DCT:
	{
		typedef multi_array<double, 2> array_t;
		array_t A(extents[5][5]);
		array_t A_out(extents[5][5]);

		auto dct = plan_dct(A, A_out);
		auto idct = plan_idct(A_out, A);

		iota(A, 1);
		array_t A_orig(A);

		dct();
		idct();

		cout << A_orig << endl << A_out << endl << A << endl;

		// Scaling:
		auto A_i = all_elements(A), O_i = all_elements(A_orig);
		assert(equal(A_i.begin(), A_i.end(), O_i.begin(),
			[](double a, double b){return abs(a - b * 100) < 1e-6; }
		));

		// Inner output is empty:
		auto out_inner = A_out[indices[range(1,5)][range(1,5)]];
		cout << out_inner << endl;
		for(auto e : all_elements(out_inner))
			assert( abs(e) < 1e-6 );
	}

#if 0
	// DCT of a view
	{
		multi_array<double, 4> a(extents[2][5][5][10]);
		assert(c_equal(a.index_bases(), {0, 0, 0, 0}));
		assert(c_equal(a.strides(), {250, 50, 10, 1}));
		assert(c_equal(a.shape(), {2, 5, 5, 10}));

		auto sub = a[indices[1][range(0,5,2)][range(0,5,3)][range(1,7,3)]];
		assert(c_equal(sub.index_bases(), {0, 0, 0}));
		assert(c_equal(sub.strides(), {100, 30, 3}));
		assert(c_equal(sub.shape(), {3, 2, 2}));
		
		multi_array<double, 3> out;
		try {
			auto dct = plan_dct(sub, out);
			assert(false);
		} catch(invalid_argument) {
			// unsupported stride.
			assert(true);
		}

		auto sub2 = a[indices[1][range(0,5,2)][range(0,5,2)][range(1,7,2)]];
		assert(c_equal(sub2.index_bases(), {0, 0, 0}));
		assert(c_equal(sub2.strides(), {100, 20, 2}));
		assert(c_equal(sub2.shape(), {3, 3, 3}));

		auto dct = plan_dct(sub2, out);
		auto idct = plan_dct(out, sub2);
		iota(sub2, 1);
		cout << sub2 << endl << out << endl;
		//dct();
		//idct();
		cout << sub2 << endl << out << endl;


//		iota(sub, 1);
//		cout << sub << endl;

//		multi_array<double, 3> out;
//		auto dct = plan_dct(sub, out);
//		auto idct = plan_idct(out, sub);
//		iota(sub, 1);
//		cout << sub << endl;
//		cout << a << endl;
	}
#endif
	
	return 0;
}
