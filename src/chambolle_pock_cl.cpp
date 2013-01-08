#include "chambolle_pock.h"
#if HAVE_OPENCL

#include "opencl_multi_array.h"

using namespace std;
using namespace boost;
using namespace cl;

typedef complex<float> float2;


#define debug_(bufname, desc, type, fun) \
	if(debug) {\
		multi_array<type, 2> __data(size); \
		ctx.wait(); cerr << "[DEBUG] "; ctx(bufname >> __data).then().resume(); \
		debug_log.push_back(debug_state(fun, #bufname ": " desc)); \
	}

#define debug_r(bufname, desc) debug_(bufname, desc, float, __data)
#define debug_c(bufname, desc) debug_(bufname, desc, float2, real(__data))

#define kernel(name) \
	auto name = prog[#name]


using namespace boost;

extern const char* chambolle_pock_kernels;


multi_array<float, 2> chambolle_pock::run_cl(const multi_array<float, 2> &x_in) {
	assert(sigma > 0);

	const size_t N = constraints.size();

	const auto size = extents_of(x_in);
	const vector<size_t> size_2d{x_in.shape()[0], x_in.shape()[1]};
	const size_t size_1d = size_2d[0] * size_2d[1];
	const unsigned int width = size[1], height = size[0];

	// fft scale factor (AMDFFT scales automatically!)
	// const float scale = 1;

	context ctx;
	auto prog = ctx.compile(chambolle_pock_kernels);
	kernel(real2complex).size({size_1d});
	kernel(pad);
	kernel(conjugate_transpose_pad);
	kernel(reduce_max_norm).size({size_1d});
	kernel(complex_mul).size({size_1d});
	kernel(add2v).size({size_1d});
	kernel(mul_add3vs).size({size_1d});
	kernel(mul_add2vs).size({size_1d});
	kernel(mul_add_clamp).size({size_1d});
	kernel(sub).size({size_1d});
	kernel(random_fill).size({size_1d});
	kernel(reduce_max_abs).size({1});

	cl::fft fft(size_2d);

	buffer
		/*float2*/fft_k[N], // >->
		/*float2*/fft_conj_k[N], // >->
		/*float2*/y[N]; // >-
	buffer
		x(sizeof(float) * size_1d),
		max_norm(sizeof(float) * size_1d),
		new_x(sizeof(float) * size_1d),
		bar_x(sizeof(float2) * size_1d), // >-
		w(sizeof(float) * size_1d),
		convolved(sizeof(float2) * size_1d), // ->
		Y(sizeof(float) * size_1d),
		fft_bar_x(sizeof(float2) * size_1d), // ->
		fft_conv(sizeof(float2) * size_1d), // >-, ->
		result(sizeof(float) * size_1d);

	// Upload input data.
	auto seq = ctx( x << x_in );
	debug_r(x, "initial");
	seq = seq.then()( Y = x );
	debug_r(Y, "keep");
	seq = seq.then()( real2complex(x, bar_x) );

	// Preprocess the kernels, i.e. pad and apply fourier transform.
	float max_norm_v[N];
	for(size_t i = 0 ; i < N ; i++) {
		// init y.
		seq = seq.then()( y[i].fill(float2(0), size_1d) );

		// Get kernel
		auto k = constraints[i].get_k(x_in);
		const vector<size_t> k_size{k.shape()[0], k.shape()[1]};

		// Pad and transform kernel
		buffer k_real;
		seq = seq.then()( k_real << k );
		seq = seq.then()( fft_k[i].fill(float2(0), size_1d) );
		seq = seq.then()( pad(k_real, fft_k[i], width).size(k_size) );
		debug_c(fft_k[i], "kernel padded");
		seq = seq.then()( fft.forward(fft_k[i]) );
		debug_c(fft_k[i], "forward fft");

		// Calculate max norm of transformed kernel
		seq = seq.then()( reduce_max_norm(fft_k[i], max_norm) );
		seq = seq.then()( max_norm >> max_norm_v[i] );

		// Conjugate pad and transform kernel
		seq = seq.then()( fft_conj_k[i].fill(float2(0), size_1d) );
		seq = seq.then()( conjugate_transpose_pad(k_real, fft_conj_k[i], width, height).size(k_size) );
		debug_c(fft_conj_k[i], "conjtransp padded");
		seq = seq.then()( fft.forward(fft_conj_k[i]) );
		debug_c(fft_conj_k[i], "forward fft");
	}

	seq.then().resume();
	float total_norm = 0;
	for(size_t i = 0 ; i < N ; i++)
		total_norm += max_norm_v[i];
	// Adjust sigma with norm.
	sigma /= tau * total_norm;
	cerr << "total norm = " << total_norm << endl;

	// If needed, calculate `q` value.
	const float q = cached_q(size, [&]{
		cerr << "monte carlo sim, " << monte_carlo_steps << " steps" << endl;
		vector<float> qs;
		buffer
			data(sizeof(float2) * size_1d),
			convolved(sizeof(float2) * size_1d),
			absmax(sizeof(float) * size_1d);
		random_device dev;
		mt19937 gen(dev());
		for(int i = 0 ; i < monte_carlo_steps ; i++) {
			seq = seq.then()( random_fill(data, cl_uint(size_1d), cl_uint(gen()), float(sigma)) );
			debug_c(data, "random data");
			seq = seq.then()( fft.forward(data) );
			debug_c(data, "forward fft'd");
			float that_q = 0;
			for(size_t j = 0 ; j < N ; j++) {
				seq = seq.then()( complex_mul(convolved, data, fft_k[j]) );
				seq = seq.then()( fft.backward(convolved) );
				debug_c(convolved, "convolved");
				seq = seq.then()( reduce_max_abs(convolved, cl_uint(size_1d), absmax) );
				float kernel_q = -1;
				seq = seq.then()( absmax >> kernel_q );
				seq.then().resume();
				that_q = max(that_q, kernel_q);
				cerr << "step " << i << '/' << monte_carlo_steps << " kernel" << j << " kernel_q=" << kernel_q << endl;;
			}
			cerr << "step " << i << " q=" << that_q << endl;
			qs.push_back(that_q);
		}
		seq.then().resume();
		return qs;
	});
	cerr << "q = " << q << "   " << endl;


	// Repeat until good enough.
	for(int n = 0 ; n < max_steps ; n++) {
		// reset accumulator
		seq = seq.then()( w = 0.0f );

		// transform bar_x for convolutions
		seq = seq.then()( fft.forward(bar_x, fft_bar_x) );
		debug_c(fft_bar_x, "forward fft bar_x");

		// for each constraint
		for(size_t i = 0 ; i < N ; i++) {
			// convolve bar_x with kernel
			seq = seq.then()( complex_mul(fft_conv, fft_k[i], fft_bar_x) );
			debug_c(fft_conv, "kernel * bar_x");
			seq = seq.then()( fft.backward(fft_conv, convolved) );
			debug_c(convolved, "backward fft");

			// calculate new y_i
			seq = seq.then()( mul_add_clamp(y[i], y[i], convolved,
				float(sigma), float(-q * sigma), float(q * sigma)) );
			debug_c(y[i], "new y");

			// convolve y_i with conjugate transpose of kernel
			seq = seq.then()( fft.forward(y[i], fft_conv) );
			debug_c(fft_conv, "y[i] forward fft");
			seq = seq.then()( complex_mul(fft_conv, fft_conv, fft_conj_k[i]) );
			debug_c(fft_conv, "kernel' * y[i]");
			seq = seq.then()( fft.backward(fft_conv, convolved) );
			debug_c(convolved, "backward fft");
			
			// accumulate
			seq = seq.then()( add2v(w, w, convolved) );
		}

		// new x
		const float tau_tau = tau / (1 + tau);
		seq = seq.then()( mul_add3vs(new_x, x, float(1 / (1 + tau)), w, -tau_tau, Y, tau_tau) );
		debug_r(new_x, "x - w + Y");

		// theta
		const float theta = 1 / sqrt(1 + 2 * tau * gamma);

		tau *= theta;
		sigma /= theta;

		// new bar_x
		seq = seq.then()( mul_add2vs(bar_x, new_x, theta + 1, x, -theta) );
		debug_c(bar_x, "x - old_x");
		seq = seq.then()( x = new_x );
	}

	debug_r(Y, "original");
	debug_r(x, "result");
	multi_array<float, 2> result_a(size);
	seq = seq.then()( sub(result, Y, x) );
	debug_r(result, "reconstruction");
	seq = seq.then()( result >> result_a );
	seq.then().resume();

	return result_a;
}

#endif
