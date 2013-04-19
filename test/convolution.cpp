#include "multi_array_fft.h"
#include "multi_array_operators.h"
#include <iostream>
#include "multi_array_io.h"
#include "watch.h"

using namespace std;
using namespace boost;
using namespace mimas;

typedef float T;
typedef complex<float> T2;
typedef multi_array<T, 2> A;
typedef multi_array<T2, 2> A2;

size_t rep = 10;


void dir_conv(const A &in, A &out, size_t h, bool adj) {
	watch w;
	for(size_t r = 0 ; r < rep ; r++) {
		const T scale = 1.0 / (h * h);
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
	auto t = w() / rep;
	cout << "dir: " << t << "s/run" << endl;
}

void fft_conv(const A &in, A &out, size_t h, bool adj) {
	watch w_once;
	fftw::plan<T, T2, 2> fft(in.shape());
	fftw::plan<T2, T, 2> ifft(in.shape());
	const size_t s0 = in.shape()[0], s1 = in.shape()[1];
	A2 f_in(extents[s0][s1/2+1]);
	fft(in, f_in);
	double t_once = w_once();
	watch w_kern;
	A k(in);
	A2 f_k(f_in);
	fill(k, 0);
	for(size_t i0 = 0 ; i0 < h ; i0++)
		for(size_t i1 = 0 ; i1 < h ; i1++) {
			if(adj) k[i0][i1] = 1;
			else k[(s0 - i0) % s0][(s1 - i1) % s1] = 1;
		}
	fft(k, f_k);
	double t_kern = w_kern();
	watch w_run;
	for(size_t r = 0 ; r < rep ; r++) {
		const T scale = 1.0 / (s0 * s1) / (h * h);
		for(size_t i0 = 0 ; i0 < s0 ; i0++)
			for(size_t i1 = 0 ; i1 < s1/2+1 ; i1++)
				f_in[i0][i1] *= f_k[i0][i1] * scale;
		ifft(f_in, out);
	}
	double t_run = w_run() / rep;
	cout << "fft: " << t_once << "s + " << t_kern << "s/kern + " << t_run << "s/run" << endl;
}

void sat_conv(const A &in, A &out, size_t h, bool adj) {
	watch w_init;
	const size_t s0 = in.shape()[0], s1 = in.shape()[1];
	A sat(in);
	for(size_t i0 = 0 ; i0 < s0 ; i0++)
		for(size_t i1 = 0 ; i1 < s1 ; i1++)
			sat[i0][i1] = in[i0][i1]
				+ (i0 > 0 ? sat[i0 - 1][i1] : 0)
				+ (i1 > 0 ? sat[i0][i1 - 1] : 0)
				- (i0 > 0 && i1 > 0 ? sat[i0 - 1][i1 - 1] : 0);
	double t_init = w_init();
	const auto box_sum = [&](size_t i0, size_t i1, size_t j0, size_t j1) {
		// i0..j0 i1..j1 inclusive.
		// corner values.
		const auto a = sat[i0][i1], b = sat[i0][j1], c = sat[j0][i1], d = sat[j0][j1];
		// projections to right
		const auto ar = sat[i0][s1-1], cr = sat[j0][s1-1];
		// projections to bottom
		const auto ab = sat[s0-1][i1], bb = sat[s0-1][j1];
		// lower right corner
		const auto rb = sat[s0-1][s1-1];
		// cases.
		if(i0 <= j0) {
			if(i1 <= j1) // normal.
				return a - b - c + d;
			else // right wrap
				return (d - b) + (a - ar - c + cr);
		} else {
			if(i1 <= j1) // bottom wrap
				return (a - ab - b + bb) + (d - c);
			else // bottom-right wrap
				return d + (cr - c) + (bb - b) + (a - ab - ar + rb);
		}
	};
	watch w_run;
	for(size_t r = 0 ; r < rep ; r++) {
		const T scale = 1.0 / (h * h);
		for(size_t i0 = 0 ; i0 < s0 ; i0++)
			for(size_t i1 = 0 ; i1 < s1 ; i1++) {
				if(adj)
					out[i0][i1] = scale * box_sum(
						(i0 + s0 - h) % s0,
						(i1 + s1 - h) % s1,
						i0,
						i1
					);
				else
					out[i0][i1] = scale * box_sum(
						(i0 + s0 - 1) % s0,
						(i1 + s1 - 1) % s1,
						(i0 + h - 1) % s0,
						(i1 + h - 1) % s1
					);
			}
	}
	double t_run = w_run() / rep;
	cout << "sat: " << t_init << "s + " << t_run << "s/run" << endl;
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

template<class F>
void check_adj(F conv, const A &x, const A &y) {
	size_t h = 40;
	A kx(x), aky(x);
	conv(x, kx, h, false);
	conv(y, aky, h, true);
	T l = dot(kx, y), r = dot(x, aky);
	cout << l << " - " << r << "  = " << abs(l - r) << endl;
}

int main(int argc, char **argv) {
	if(argc == 2) { // use input image.
		Gtk::Main main(argc, argv);
		auto in_pb = Gdk::Pixbuf::create_from_file(argv[1]);
		A in = pixbuf_to_multi_array(in_pb);
		A out1(in), out2(in), out3(in);
		size_t h = 20;
		bool adj = true;
		dir_conv(in, out1, h, adj);
		fft_conv(in, out2, h, adj);
		sat_conv(in, out3, h, adj);
		cout << max(absolute(out1 - out2)) << endl;
		cout << max(absolute(out1 - out3)) << endl;
		multi_array_to_pixbuf(out1)->save("out1.png", "png");
		multi_array_to_pixbuf(out2)->save("out2.png", "png");
		multi_array_to_pixbuf(out3)->save("out3.png", "png");
	} else { // test: dot(adj(K) * X, Y) = dot(X, K * Y)
		A x(extents[512][512]), y(x);
		fillrandom(x);
		fillrandom(y);
		check_adj(dir_conv, x, y);
		check_adj(fft_conv, x, y);
		check_adj(sat_conv, x, y);
	}
	return EXIT_SUCCESS;
	
}

