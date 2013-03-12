#include "chambolle_pock.h"
#if HAVE_OPENCL

#include <vexcl/vexcl.hpp>
#include <vexcl/fft.hpp>
#include <vexcl/random.hpp>

#define debug_(bufname, desc, type, fun) \
	if(debug) {\
	}
		//boost::multi_array<type, 2> __data(size); \
		copy(bufname, __data.data()); \
		debug_log.push_back(debug_state(fun, #bufname ": " desc)); \
	}

#define debug_r(bufname, desc) debug_(bufname, desc, float, __data)
#define debug_c(bufname, desc) debug_(bufname, desc, cl_float2, mimas::real(__data))


boost::multi_array<float, 2> chambolle_pock::run_cl(const boost::multi_array<float, 2> &x_in) {
	using namespace vex;

	const size_t N = constraints.size();

	const auto size = extents_of(x_in);
	const std::vector<size_t> size_2d{x_in.shape()[0], x_in.shape()[1]};
	const size_t size_1d = size_2d[0] * size_2d[1];
	const unsigned int width = size[1], height = size[0];

	Context ctx(Filter::Env && Filter::Count(1));
	std::cout << ctx << std::endl;

	// Functions.
	FFT<cl_float, cl_float2> fft(ctx.queue(), size_2d);
	FFT<cl_float2, cl_float> ifft(ctx.queue(), size_2d, inverse);
	Reductor<cl_float, MAX> max(ctx.queue());

	VEX_FUNCTION(norm, cl_float(cl_float2),
		"return dot(prm1, prm1);");
	VEX_FUNCTION(complex_mul, cl_float2(cl_float2, cl_float2),
		"return (float2)("
		"prm1.x * prm2.x - prm1.y * prm2.y,"
		"prm1.x * prm2.y + prm1.y * prm2.x);");
	VEX_FUNCTION(soft_clamp, cl_float(cl_float, cl_float),
		"if(prm1 < -prm2) return prm1 + prm2;"
		"if(prm1 > prm2) return prm1 - prm2;"
		"return 0;");

	vector<cl_float> x(ctx.queue(), size_1d, x_in.data()),
		max_norm(ctx.queue(), size_1d),
		new_x(ctx.queue(), size_1d),
		bar_x(ctx.queue(), size_1d),
		w(ctx.queue(), size_1d),
		Y(ctx.queue(), size_1d, x_in.data()),
		result(ctx.queue(), size_1d),
		convolved(ctx.queue(), size_1d);
	vector<cl_float2>
		fft_bar_x(ctx.queue(), size_1d),
		fft_conv(ctx.queue(), size_1d);
	std::vector<vector<cl_float>> y;
	std::vector<vector<cl_float2>> fft_k, fft_conj_k;

	// Upload input data (on creation)
	debug_r(x, "initial");
	debug_r(Y, "keep");

	bar_x = x;

	// Preprocess the kernels, i.e. pad and apply fourier transform.
	float total_norm = 0;
	for(size_t i = 0 ; i < N ; i++) {
		// init y.
		y.emplace_back(ctx.queue(), size_1d);
		y[i] = 0.0f;

		// Get kernel
		auto k = constraints[i].get_k(x_in);
		const std::vector<size_t> k_size{k.shape()[0], k.shape()[1]};

		// Pad and transform kernel
		{
			boost::multi_array<float, 2> padded_kernel(size_2d);
			fill(padded_kernel, 0);
			kernel_pad(k, padded_kernel);
			vector<cl_float> k_real(ctx.queue(), size_1d, padded_kernel.data());
			fft_k.emplace_back(ctx.queue(), size_1d);
			debug_r(k_real, "kernel padded");
			fft_k[i] = fft(k_real);
			debug_c(fft_k[i], "forward fft");
		}

		// Calculate max norm of transformed kernel
		total_norm += max(norm(fft_k[i]));

		// Conjugate pad and transform kernel
		{
			boost::multi_array<float, 2> padded_kernel(size_2d);
			auto t = conjugate_transpose(k);
			fill(padded_kernel, 0);
			kernel_pad(t, padded_kernel, true);
			vector<cl_float> k_real(ctx.queue(), size_1d, padded_kernel.data());
			debug_r(k_real, "conjtransp padded");
			fft_conj_k.emplace_back(ctx.queue(), size_1d);
			fft_conj_k[i] = fft(k_real);
			debug_c(fft_conj_k[i], "forward fft");
		}
	}

	// Adjust sigma with norm.
	sigma /= tau * total_norm;
	std::cerr << "total norm = " << total_norm << std::endl;

	// If needed, calculate `q/sigma` value.
	float q = sigma * cached_q(size, [&]{
		std::cerr << "monte carlo sim, " << monte_carlo_steps << " steps" << std::endl;
#define DUMP_MC_DETAILS
#ifdef DUMP_MC_DETAILS
		std::ofstream d("monte-carlo-debug.dat");
		for(auto c : constraints) d << c.expr << '\t';
		d << "max\n";
		d << std::setprecision(12) << std::scientific;
#endif
#ifdef DUMP_RANDOM_NUMBERS
		std::ofstream d2("monte-carlo-randoms.dat");
		d2 << std::setprecision(12) << std::scientific;
#endif
		std::vector<float> qs;
		vector<cl_float> data(ctx.queue(), size_1d);
		vector<cl_float> convolved(ctx.queue(), size_1d);
		vector<cl_float2> data_fft(ctx.queue(), size_1d);
		vector<cl_float2> multiplied(ctx.queue(), size_1d);
		vector<cl_float> absmax(ctx.queue(), size_1d);
		RandomNormal<cl_float> random;
		std::random_device dev;
		std::mt19937 gen(dev());
		for(int i = 0 ; i < monte_carlo_steps ; i++) {
			data = random(element_index(), gen());
#ifdef DUMP_RANDOM_NUMBERS
			std::vector<float> data_h(size_1d);
			copy(data, data_h);
			for(auto x : data_h) d2 << x << '\n';
#endif
			debug_c(data, "random data");
			data_fft = fft(data);
			debug_c(data, "forward fft'd");
			std::vector<float> k_qs;
			for(size_t j = 0 ; j < N ; j++) {
				multiplied = complex_mul(data_fft, fft_k[j]);
				convolved = ifft(multiplied);
				debug_r(convolved, "convolved");
				k_qs.push_back(max(fabs(convolved)));
			}
			float max_q = *max_element(k_qs.begin(), k_qs.end());

#ifdef DUMP_MC_DETAILS
			for(auto q : k_qs) d << q << '\t';
			d << max_q << std::endl;
#endif
			if(i % 100 == 0)
				std::cerr << "step " << i << " q=" << max_q << std::endl;
			qs.push_back(max_q);
		}
		return qs;
	});
	std::cerr << "q = " << q << "   " << std::endl;


	// Repeat until good enough.
	for(int n = 0 ; n < max_steps ; n++) {
		// reset accumulator
		w = 0.0f;

		// transform bar_x for convolutions
		fft_bar_x = fft(bar_x);
		debug_c(fft_bar_x, "forward fft bar_x");

		// for each constraint
		for(size_t i = 0 ; i < N ; i++) {
			// convolve bar_x with kernel
			fft_conv = complex_mul(fft_k[i], fft_bar_x);
			debug_c(fft_conv, "kernel * bar_x");
			convolved = ifft(fft_conv);
			debug_c(convolved, "backward fft");

			// calculate new y_i
			y[i] = soft_clamp(y[i] + convolved * float(sigma), float(q * sigma));
			debug_c(y[i], "new y");

			// convolve y_i with conjugate transpose of kernel
			fft_conv = fft(y[i]);
			debug_c(fft_conv, "y[i] forward fft");
			fft_conv = complex_mul(fft_conv, fft_conj_k[i]);
			debug_c(fft_conv, "kernel' * y[i]");
			convolved = ifft(fft_conv);
			debug_c(convolved, "backward fft");
			
			// accumulate
			w += convolved;
		}

		// new x
		const float tau_tau = tau / (1 + tau);
		new_x = x * float(1 / (1 + tau)) - w * tau_tau + Y * tau_tau;
		debug_r(new_x, "x - w + Y");

		// theta
		const float theta = 1 / sqrt(1 + 2 * tau * gamma);

		tau *= theta;
		sigma /= theta;

		// new bar_x
		bar_x = new_x * (theta + 1) - x * theta;
		debug_c(bar_x, "x - old_x");
		x = new_x;
	}

	debug_r(Y, "original");
	debug_r(x, "result");
	result = Y - x;
	boost::multi_array<float, 2> result_a(size);
	copy(result, result_a.data());
	debug_r(result, "reconstruction");

	return result_a;
}

#endif
