#include <cassert>
#include "multi_array_operators.h"
#include "multi_array_print.h"
#include "multi_array_matrix.h"

#include <boost/numeric/ublas/io.hpp>

using namespace std;
using namespace boost;
using namespace boost::numeric::ublas;


int main(int, char**) {
	// 2D matrix
	typedef multi_array<double, 2> array_t;
	typedef array_t::index index_t;

	array_t A( extents[3][4] );
	iota(A, 0);

	cout << A << endl;

	auto a = make_matrix(A);
	cout << a << endl;
	cout << typeid(a).name() << endl;

	a *= 2;
	cout << a << endl;
	cout << A << endl; // A changed



	// 3D matrix slice
	typedef multi_array<double, 3> array3;
	typedef array3::index index3;
	array3 B( extents[3][4][5] );
	iota(B, 0);

	cout << B << endl;
	cout << typeid(B).name() << endl;

	// modifying sub_array modifies original:
	auto B1 = B[0];
	cout << array_t(B1) << endl;
	cout << make_matrix(B1) << endl;
	make_matrix(B1) *= 2;
	cout << make_matrix(B1) << endl; // changed

	// modifying view modifies original:
	typedef multi_array_types::index_range range;
	array3::index_gen gen;
	auto B2 = B[ gen[range(0, 3)][1][range(0,5,2)] ];
	cout << array_t(B2) << endl;
	cout << make_matrix(B2) << endl;
	make_matrix(B2) *= 2;
	cout << make_matrix(B2) << endl; // changed





	// no support for immutable arrays though.
	const auto C(B);
	const auto C1 = C[0];
	const auto C2 = C[ gen[range(0, 3)][1][range(0,5,2)] ];
	//make_matrix(C); // nope
	//make_matrix(C1); // nope
	//make_matrix(C2); // nope



	
	// origin
	array_t D( extents[3][3] );
	D.reindex(array<index_t, 2>{{-1, -1}});
	D[0][0] = 1;
	cout << D << endl;
	cout << make_matrix(D) << endl;
}

