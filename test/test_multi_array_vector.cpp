#include <cassert>
#include "multi_array.h"
#include "multi_array_vector.h"
#include "multi_array_print.h"
#include "multi_array_compare.h"

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace std;
using namespace boost;
using namespace boost::numeric::ublas;




int main(int, char**) {
	// ublas vector:
	typedef boost::numeric::ublas::vector<double> dvec; 
	dvec v(3);
	for(size_t i = 0 ; i != v.size() ; i++)
		v(i) = i;

	assert(range_equal(v, {0, 1, 2}));
	assert( norm_1(v) == 3 );

	// multi_array adapted vector:
	multi_array<double, 1> u( extents[3] );
	for(size_t i = 0 ; i != u.size() ; i++)
		u[i] = i;

	assert( norm_1(make_vector(u)) == 3 );

	// no copying:
	make_vector(u) *= 2;
	assert(multi_equal(u, {0, 2, 4}));

	// combined:
	dvec w = make_vector(u) + v;
	assert(range_equal(w, {0, 3, 6})); // sum
	assert(multi_equal(u, {0, 2, 4})); // unchanged
	assert(range_equal(v, {0, 1, 2})); // unchanged

	// resizing
	auto ad = make_vector(u);
	assert(multi_equal(u, {0, 2, 4}));
	assert(range_equal(ad, {0, 2, 4}));

	ad.resize(2);
	assert(range_equal(ad, {0, 2}));
	assert(multi_equal(u, {0, 2}));

	ad.resize(4);
	assert(multi_equal(u, {0, 2, 0, 0}));
	assert(range_equal(ad, {0, 2, 0, 0}));

	// vector of a view
	{
		multi_array<double, 2> multi(extents[4][9]);
		iota(multi, 1);
		auto v = multi[indices[2][boost::range(1, 7, 2)]];
		assert(multi_equal(v, {20, 22, 24}));

		auto u = make_vector(v);
		assert(range_equal(u, {20, 22, 24}));
		u[1] = 999;
		assert(multi[2][3] == 999);
	}

	return 0;
}
