#include "chambolle_pock.h"
#include "gil_io.h"
#include <iomanip>

using namespace mimas;
using namespace std;
using namespace boost;

template<class A>
void normalize(A &in) {
	auto lo = min(in), hi = max(in);
	cerr << " lo=" << lo << " hi=" << hi << endl;
	if(lo > 0) lo = 0;
	if(hi < 1) hi = 1;
//	lo = -1; hi = 1;
	in -= lo;
	in /= hi - lo;
}

multi_array<float, 2> gauss(size_t w, size_t h, float sigma) {
	multi_array<float, 2> out(extents[h][w]);
	const float den = -1 / (2 * pow(sigma, 2));
	float x0 = w/2.0, y0 = h/2.0;
	for(size_t y = 0 ; y < h ; y++)
		for(size_t x = 0 ; x < w ; x++)
			out[y][x] = exp( den * (pow(x - x0, 2) + pow(y - y0, 2)) );
	return out;
}


int main(int argc, char **argv) {
	if(argc != 3) return EXIT_FAILURE;

	// read input
	auto x = read_image(argv[1]);
	const size_t w = x.shape()[0], h = x.shape()[1];
	x *= 2.0f;
	x -= 1.0f;

	// constraints
	vector<constraint> cs;

	for(size_t k_size = 1 ; k_size <= 10 ; k_size++) {
		auto k = gauss(w, h, k_size / 5.0);
		k /= sum(k);
		cs.push_back(constraint{-1, 1, x, k});
	}

	// run
	const float tau = 1e10;
	const float sigma = 1.0f / tau;
	const float gamma = 1.0f;

	auto bar_x = chambolle_pock(tau, sigma, gamma, x, cs, [](const multi_array<float, 2> &x, string name){
		cerr << name;
		multi_array<float, 2> xn = x;
		normalize(xn);
		write_image(name + ".png", xn);
	});


	normalize(bar_x);
	write_image(argv[2], bar_x);

	return EXIT_SUCCESS;
}
