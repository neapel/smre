#include "chambolle_pock.h"
#include "multi_array_fft.h"

using namespace std;
using namespace mimas;
using namespace boost;

inline float clamp(float x, float low, float high) {
	if(x < low) return low;
	if(x > high) return high;
	return x;
}

multi_array<float, 2> chambolle_pock(size_t max_steps, float tau, float sigma, float gamma, multi_array<float, 2> &x, vector<constraint> &cons, debug_f debug) {
	assert(sigma > 0);

	const size_t N = cons.size();

	const auto size = extents_of(x);
	const auto width = size[0], height = size[1];
	const auto c_width = width / 2 + 1, c_height = height;
	const auto c_size = extents[c_height][c_width];

	// fftw scale factor
	const float scale = 1 / sqrt(width * height);

	// arrays
	multi_array<float, 2> bar_x = x, w(size), convolved(size), dx(size), Y = x, padded_kernel(size);
	multi_array<complex<float>, 2> fft_bar_x(c_size), fft_conv(c_size);

	// Preprocess the kernels, i.e. pad and apply fourier transform.
	multi_array<complex<float>, 2> fft_k[N], fft_conj_k[N];
	for(size_t i = 0 ; i < N ; i++) {
		// normalize kernel to sum=1
		cons[i].k /= sum(cons[i].k);

		// Pad and transform kernel
		fft_k[i].resize(c_size);
		kernel_pad(cons[i].k, padded_kernel);
		fftw::forward(padded_kernel, fft_k[i])();

		// Conjugate pad and transform kernel
		fft_conj_k[i].resize(c_size);
		auto t = conjugate_transpose(cons[i].k);
		kernel_pad(t, padded_kernel);
		fftw::forward(padded_kernel, fft_conj_k[i])();
	}

	// Repeat until good enough.
	for(size_t n = 0 ; n < max_steps ; n++) {
		// reset accumulator
		fill(w, 0);

		// transform bar_x for convolutions
		fftw::forward(bar_x, fft_bar_x)();

		// for each constraint
		for(size_t i = 0 ; i < N ; i++) {
			// convolve bar_x with kernel
			for(size_t iy = 0 ; iy < c_height ; iy++)
				for(size_t ix = 0 ; ix < c_width ; ix++)
					fft_conv[iy][ix] = fft_k[i][iy][ix] * fft_bar_x[iy][ix] * scale;
			fftw::backward(fft_conv, convolved)();
			convolved *= scale;
			debug(convolved, "convolved", n, i, tau, sigma, -1);

			// calculate new y_i
			for(size_t iy = 0 ; iy < height ; iy++)
				for(size_t ix = 0 ; ix < width ; ix++)
					cons[i].y[iy][ix] = (cons[i].y[iy][ix] + sigma * convolved[iy][ix])
					 - sigma * clamp(cons[i].y[iy][ix] / sigma + convolved[iy][ix], cons[i].a, cons[i].b);
			debug(cons[i].y, "y", n, i, tau, sigma, -1);

			// convolve y_i with conjugate transpose of kernel
			fftw::forward(cons[i].y, fft_conv)();
			for(size_t iy = 0 ; iy < c_height ; iy++)
				for(size_t ix = 0 ; ix < c_width ; ix++)
					fft_conv[iy][ix] *= fft_conj_k[i][iy][ix] * scale;
			fftw::backward(fft_conv, convolved)();
			convolved *= scale;

			// accumulate
			w += convolved;
		}
		debug(w, "w", n, -1, tau, sigma, -1);

		auto old_x = x;

		// new x
		for(size_t iy = 0 ; iy < height ; iy++)
			for(size_t ix = 0 ; ix < width ; ix++)
				x[iy][ix] = (x[iy][ix] - tau * w[iy][ix] + tau * Y[iy][ix]) / (1 + tau);
		debug(x, "x", n, -1, tau, sigma, -1);

		// theta
		const float theta = 1 / sqrt(1 + 2 * tau * gamma);

		tau *= theta;
		sigma /= theta;

		// new bar_x
		for(size_t iy = 0 ; iy < height ; iy++)
			for(size_t ix = 0 ; ix < width ; ix++)
				bar_x[iy][ix] = x[iy][ix] + theta * (x[iy][ix] - old_x[iy][ix]);
		debug(bar_x, "x̄", n, -1, tau, sigma, theta);
	}

	return bar_x;
}

