#include "chambolle_pock.h"
#include "multi_array_fft.h"

using namespace std;
using namespace mimas;
using namespace boost;

inline float clamp(float x, float low, float high) {
	if(x < low) return x-low;
	if(x > high) return x-high;
	else return 0;
	return x;
}

#define debug(...) \
	if(debug) debug_log.push_back(debug_state{__VA_ARGS__});

multi_array<float, 2> chambolle_pock::run(const multi_array<float, 2> &x_in) {
	assert(sigma > 0);

	auto x = x_in;

	const size_t N = constraints.size();

	const auto size = extents_of(x);
	const auto width = size[0], height = size[1];
	const auto c_width = width, c_height = height/2+1;
	const auto c_size = extents[c_width][c_height];

	// fftw scale factor
	const float scale = 1.0f / (width*height);

	// arrays
	multi_array<float, 2> bar_x = x, w(size), convolved(size), dx(size), Y = x, padded_kernel(size);
	multi_array<complex<float>, 2> fft_bar_x(c_size), fft_conv(c_size);
	vector<multi_array<float, 2>> y;

	// Preprocess the kernels, i.e. pad and apply fourier transform.
	multi_array<complex<float>, 2> fft_k[N], fft_conj_k[N];
	float total_norm = 0;
	for(size_t i = 0 ; i < N ; i++) {
		// Get kernel
		auto k = constraints[i]->get_k(x_in);

		// Pad and transform kernel
		fft_k[i].resize(c_size);
		kernel_pad(k, padded_kernel);
		fftw::forward(padded_kernel, fft_k[i])();

		// Calculate max norm of transformed kernel
		float max_norm = 0;
		for(auto row : fft_k[i]) for(auto value : row)
			max_norm = max(norm(value), max_norm);
		cout << "constraint " << i << " max norm=" << max_norm << endl;
		total_norm += max_norm;

		// Conjugate pad and transform kernel
		fft_conj_k[i].resize(c_size);
		auto t = conjugate_transpose(k);
		fill(padded_kernel,0);
		kernel_pad(t, padded_kernel,1);
		fftw::forward(padded_kernel, fft_conj_k[i])();

		y.push_back(multi_array<float, 2>(size));
	}
	// Adjust sigma with norm.
	sigma /= tau * total_norm;

	// Repeat until good enough.
	for(size_t n = 0 ; n < max_steps ; n++) {
		// reset accumulator
		fill(w, 0);

		// transform bar_x for convolutions
		fftw::forward(bar_x, fft_bar_x)();

		// for each constraint
		for(size_t i = 0 ; i < N ; i++) {
			// convolve bar_x with kernel
			for(size_t ix = 0 ; ix < c_width ; ix++)
				for(size_t iy = 0 ; iy < c_height ; iy++)
					fft_conv[ix][iy] = fft_k[i][ix][iy] * fft_bar_x[ix][iy] * scale;
			fftw::backward(fft_conv, convolved)();
			debug(convolved, "convolved", (int)n, (int)i, tau, sigma, -1);

			// calculate new y_i
			for(size_t ix = 0 ; ix < width ; ix++)
				for(size_t iy = 0 ; iy < height ; iy++)
					y[i][ix][iy] = clamp(y[i][ix][iy] + sigma * convolved[ix][iy], constraints[i]->a * sigma, constraints[i]->b * sigma);
			debug(y[i], "y", (int)n, (int)i, tau, sigma, -1);

			// convolve y_i with conjugate transpose of kernel
			fftw::forward(y[i], fft_conv)();
			for(size_t ix = 0 ; ix < c_width ; ix++)
				for(size_t iy = 0 ; iy < c_height ; iy++)
					fft_conv[ix][iy] *= fft_conj_k[i][ix][iy] * scale;
			fftw::backward(fft_conv, convolved)();

			// accumulate
			w += convolved;
		}
		debug(w, "w", (int)n, -1, tau, sigma, -1);

		auto old_x = x;

		// new x
		for(size_t ix = 0 ; ix < width ; ix++)
			for(size_t iy = 0 ; iy < height ; iy++)
				x[ix][iy] = (x[ix][iy] - tau * w[ix][iy] + tau * Y[ix][iy]) / (1 + tau);
		debug(Y-x, "x", (int)n, -1, tau, sigma, -1);

		// theta
		const float theta = 1 / sqrt(1 + 2 * tau * gamma);

		tau *= theta;
		sigma /= theta;

		// new bar_x
		for(size_t ix = 0 ; ix < width ; ix++)
			for(size_t iy = 0 ; iy < height ; iy++)
				bar_x[ix][iy] = x[ix][iy] + theta * (x[ix][iy] - old_x[ix][iy]);
		debug(bar_x, "x̄", (int)n, -1, tau, sigma, theta);
	}

	return Y-x;
}

