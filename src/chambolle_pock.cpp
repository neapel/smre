#include "multi_array.h"

using namespace std;
using namespace mimas;





multi_array<float, 2> chambolle_pock(float tau, float sigma, multi_array<float, 2> &x, vector<constraint> cons) {
	assert(sigma > 0);

	const size_t N = cons.size();

	const auto size = extents_of(x);
	const auto width = size[0], height = size[1];

	multi_array<float, 2> bar_x = x, w(size), convolved(size), dx(size);
	multi_array<complex<float>, 2> fft_bar_x(size), fft_conv(size);

	// Preprocess the kernels, i.e. pad and apply fourier transform.
	multi_array<complex<float>, 2> fft_k[N], fft_conj_k[N];
	for(size_t i = 0 ; i < N ; i++) {
		// Pad and transform kernel
		fft_k.resize(size);
		kernel_pad(cons[i].k, fft_k);
		fft::forward(fft_k, fft_k)();

		// Conjugate pad and transform kernel
		fft_conj_k.resize(size);
		auto t = conjugate_transpose(cons[i].k);
		kernel_pad(t, fft_conj_k)();
		fft::forward(fft_conj_k, fft_conj_k)();
	}

	// Repeat until good enough.
	for(size_t n = 0 ; n < 10 ; n++) {
		// reset accumulator
		w = 0;

		// transform bar_x for convolutions
		fft::forward(bar_x, fft_bar_x);

		// for each constraint
		for(size_t i = 0 ; i < N ; i++) {
			// convolve bar_x with kernel
			for(size_t iy = 0 ; iy < height ; iy++)
				for(size_t ix = 0 ; ix < width ; ix++)
					fft_conv[iy][ix] = fft_k[i][iy][ix] * fft_bar_x[iy][ix];
			fft::backward(fft_conv, convolved);

			// calculate new y_i
			for(size_t iy = 0 ; iy < height ; iy++)
				for(size_t ix = 0 ; ix < width ; ix++)
					y[i][iy][ix] = (y[i][iy][ix] + sigma * convolved[iy][ix])
					             - sigma * clamp((1.0/sigma) * y[i][iy][ix] + convolved[iy][ix], a[i], b[i]);

			// convolve y_i with conjugate transpose of kernel
			fft::forward(y[i], fft_conv);
			for(size_t iy = 0 ; iy < height ; iy++)
				for(size_t ix = 0 ; ix < width ; ix++)
					fft_conv[iy][ix] *= fft_conj_k[i][iy][ix];
			fft::backward(fft_conv, convolved);

			// accumulate
			w += convolved;
		}

		// difference to accumulated
		for(size_t iy = 0 ; iy < height ; iy++)
			for(size_t ix = 0 ; ix < width ; ix++)
				dx[iy][ix] = -tau * w[iy][ix] + tau/(1 + tau) * y[iy][ix];

		// apply
		x += dx;

		// theta
		float theta = 0;
		for(size_t iy = 0 ; iy < height ; iy++)
			for(size_t ix = 0 ; ix < width ; ix++)
				theta += pow(y[iy][ix], 2);
		theta = 1 / sqrt(1 + 2 * tau * theta);

		tau *= theta;
		sigma /= theta;

		for(size_t iy = 0 ; iy < height ; iy++)
			for(size_t ix = 0 ; ix < width ; ix++)
				bar_x[iy][ix] = x[iy][ix] + theta * dx[iy][ix];
	}

	return bar_x;
}

