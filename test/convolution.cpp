#include "multi_array_fft.h"
#include "multi_array_operators.h"
#include <iostream>
#include "multi_array_io.h"
#include "convolution.h"

using namespace std;
using namespace boost;

typedef float T;
typedef multi_array<T, 2> A;

void dir_conv(const A &in, A &out, size_t h, bool adj) {
	const T scale = 1 / (M_SQRT2 * h);
	const size_t s0 = in.shape()[0], s1 = in.shape()[1];
	for(size_t i0 = 0 ; i0 < s0 ; i0++)
		for(size_t i1 = 0 ; i1 < s1 ; i1++) {
			T sum = 0;
			for(size_t d0 = 0 ; d0 < h ; d0++)
				for(size_t d1 = 0 ; d1 < h ; d1++) {
					if(adj) sum += in[(i0 + s0 - d0) % s0][(i1 + s1 - d1) % s1];
					else sum += in[(i0 + d0) % s0][(i1 + d1) % s1];
				}
			out[i0][i1] = sum * scale;
			}
}

template<class Conv>
void conv(const A &x, A &y, size_t h, bool adj) {
	auto c = new Conv(extents_of(x));
	auto i = c->_prepare_image(x);
	auto k = c->prepare_kernel(h, adj);
	c->_conv(i, k, y);
	delete c;
}

void fillrandom(A &x) {
	for(auto r : x) for(auto &v : r) v = 2.0 * rand() / RAND_MAX - 1;
}

T dot(const A &x, const A &y) {
	const size_t s0 = x.shape()[0], s1 = x.shape()[1];
	T sum = 0;
	for(size_t i0 = 0 ; i0 < s0 ; i0++)
		for(size_t i1 = 0 ; i1 < s1 ; i1++)
			sum += x[i0][i1] * y[i0][i1];
	return sum;
}

template<class Conv>
void check_adj(const A &x, const A &y, size_t h) {
	A kx(x), aky(x);
	conv<Conv>(x, kx, h, false);
	conv<Conv>(y, aky, h, true);
	T l = dot(kx, y), r = dot(x, aky);
	cout << l << " - " << r << "  = " << abs(l - r) << endl;
}

int main(int argc, char **argv) {
	vex::Context ctx(vex::Filter::Count(1));
	vex::StaticContext<>::set(ctx);

	size_t h = 20;
	if(argc == 2) { // use input image.
		Gtk::Main main(argc, argv);
		auto in_pb = Gdk::Pixbuf::create_from_file(argv[1]);
		A in = pixbuf_to_multi_array(in_pb);
		A out(in);
		bool adj = true;
		conv<gpu_fft_convolver<T>>(in, out, h, adj); multi_array_to_pixbuf(out)->save("conv_gpu_fft.png", "png");
		conv<cpu_fft_convolver<T>>(in, out, h, adj); multi_array_to_pixbuf(out)->save("conv_cpu_fft.png", "png");
		conv<gpu_sat_convolver<T>>(in, out, h, adj); multi_array_to_pixbuf(out)->save("conv_gpu_sat.png", "png");
		conv<cpu_sat_convolver<T>>(in, out, h, adj); multi_array_to_pixbuf(out)->save("conv_cpu_sat.png", "png");
	} else { // test: dot(adj(K) * X, Y) = dot(X, K * Y)
		A x(extents[512][512]), y(x);
		fillrandom(x);
		fillrandom(y);
		check_adj<gpu_fft_convolver<T>>(x, y, h);
		check_adj<cpu_fft_convolver<T>>(x, y, h);
		check_adj<gpu_sat_convolver<T>>(x, y, h);
		check_adj<cpu_sat_convolver<T>>(x, y, h);
	}
	return EXIT_SUCCESS;
	
}

