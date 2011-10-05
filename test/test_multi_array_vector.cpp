#include <cassert>
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

	assert(blas_equal(v, {0, 1, 2}));
	assert( norm_1(v) == 3 );

	// multi_array adapted vector:
	typedef multi_array<double, 1> avec;
	avec u( extents[3] );
	for(avec::size_type i = 0 ; i != u.size() ; i++)
		u[i] = i;

	assert( norm_1(make_vector(u)) == 3 );

	// no copying:
	make_vector(u) *= 2;
	assert(multi_equal(u, {0, 2, 4}));

	// combined:
	dvec w = make_vector(u) + v;
	assert(blas_equal(w, {0, 3, 6})); // sum
	assert(multi_equal(u, {0, 2, 4})); // unchanged
	assert(blas_equal(v, {0, 1, 2})); // unchanged

	// resizing
	auto ad = make_vector(u);
	assert(multi_equal(u, {0, 2, 4}));
	assert(blas_equal(ad, {0, 2, 4}));

	ad.resize(2);
	assert(blas_equal(ad, {0, 2}));
	assert(multi_equal(u, {0, 2}));

	ad.resize(4);
	assert(multi_equal(u, {0, 2, 0, 0}));
	assert(blas_equal(ad, {0, 2, 0, 0}));

	return 0;
}
