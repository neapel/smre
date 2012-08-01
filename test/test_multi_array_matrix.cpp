#include <cassert>
#include "multi_array.h"
#include "multi_array_print.h"
#include "multi_array_matrix.h"

#include <boost/numeric/ublas/io.hpp>

using namespace std;
using namespace boost;
using namespace boost::numeric::ublas;


int main(int, char**) {
	{
		multi_array<double, 2> A( extents[3][4] );
		iota(A, 0);
		assert(A[1][2] == 6);

		// derived matrix has same data
		auto a = make_matrix(A);
		assert(a(1, 2) == 6);

		// operating on matrix changes original
		a *= 2;
		assert(a(1, 2) == 12);
		assert(A[1][2] == 12);
	}


	{
		// 3D matrix slice
		multi_array<double, 3> B( extents[3][4][5] );
		iota(B, 0);
		assert(B[0][1][2] == 7);

		{
			// modifying sub_array modifies original:
			auto b = B[0];
			assert(b[1][2] == 7);
			auto m = make_matrix(b);
			assert(m(1, 2) == 7);
			m *= 2;
			assert(m(1, 2) == 14);
			m /= 2;
			assert(m(1, 2) == 7);
		}

		{
			// modifying view modifies original:
			auto b = B[ indices[boost::range(0, 3)][1][boost::range(0,5,2)] ];
			assert(b[0][1] == 7);
			auto m = make_matrix(b);
			assert(m(0, 1) == 7);
			m *= 2;
			assert(m(0, 1) == 14);
			m /= 2;
			assert(m(0, 1) == 7);
		}
	}


	{
		// no support for immutable arrays though.
	}


	{
		multi_array<double, 2> D( extents[3][3] );
		D[1][1] = 999;
		auto m = make_matrix(D);
		assert(m(1, 1) == 999);

		// shift origin after matrix creation:
		D.reindex(std::array<size_t, 2>{{-1, 0}});
		assert(D[0][1] == 999);
		// matrix doesn't support origin shift, always at 0, 0:
		assert(m(1, 1) == 999);

		// create from shifted
		auto n = make_matrix(D);
		assert(n(1, 1) == 999);
	}
}

