#include "chambolle_pock.h"
#if HAVE_OPENCL

#include "opencl_multi_array.h"

using namespace std;
using namespace boost;
using namespace cl;

typedef complex<float> float2;


#define debug_(bufname, desc, type, fun) \
	if(debug) {\
		multi_array<type, 2> data(size); \
		ctx.wait(); cerr << "[DEBUG] "; ctx(bufname >> data).then().resume(); \
		debug_log.push_back(debug_state(fun, #bufname ": " desc)); \
	}

#define debug_r(bufname, desc) debug_(bufname, desc, float, data)
#define debug_c(bufname, desc) debug_(bufname, desc, float2, real(data))

#define kernel(name) \
	auto name = prog[#name]


using namespace boost;

static string kernels = R"(
	// real image -> complex image
	__kernel void real2complex(__global float *input, __global float2 *output) {
		const size_t i = get_global_id(0);
		output[i].x = input[i];
		output[i].y = 0;
	}

	// real kernel -> padded complex kernel
	// place input in (0,0) corner of output.
	__kernel void pad(__global float *input, __global float2 *output, uint ow) {
		const size_t ky = get_global_id(0), kx = get_global_id(1);
		const size_t kh = get_global_size(0), kw = get_global_size(1);
		output[ow * ky + kx].x = input[kw * ky + kx];
	}

	// real kernel -> (conjugate) transposed padded complex kernel
	// place transposed input in (h+2, w+2) corner of output.
	__kernel void conjugate_transpose_pad(__global float *input, __global float2 *output, uint ow, uint oh) {
		const size_t ky = get_global_id(0), kx = get_global_id(1);
		const size_t kh = get_global_size(0), kw = get_global_size(1);
		const size_t dx = ow - kw + 1, dy = oh - kh + 1;
		output[ow * ((kx + dx) % ow) + ((ky + dy) % oh)].x = input[kw * kx + ky];
	}

	// find max(norm(x)) for x in data, write to output[0].
	__kernel void reduce_max_norm(__global float2 *data, __global float *output) {
		const size_t i = get_global_id(0);
		output[i] = dot(data[i], data[i]);
	}

	// a * b
	float2 _complex_mul(float2 a, float2 b) {
		float2 out;
		out.x = a.x * b.x - a.y * b.y;
		out.y = a.x * b.y + a.y * b.x;
		return out;
	}

	// Complex: out = a * b
	__kernel void complex_mul(__global float2 *out, __global float2 *a, __global float2 *b) {
		const size_t i = get_global_id(0);
		out[i] = _complex_mul(a[i], b[i]);
	}

	// Real: out = a + b.
	__kernel void add2v(__global float *out, __global float *a, __global float2 *b) {
		const size_t i = get_global_id(0);
		out[i] = a[i] + b[i].x;
	}

	// Real: out = a * as + b * bs + c * cs
	__kernel void mul_add3vs(__global float *out, __global float *a, float as, __global float *b, float bs, __global float *c, float cs) {
		const size_t i = get_global_id(0);
		out[i] = a[i] * as + b[i] * bs + c[i] * cs;
	}

	// Real: out = a * as + b * bs
	__kernel void mul_add2vs(__global float2 *out, __global float *a, float as, __global float *b, float bs) {
		const size_t i = get_global_id(0);
		out[i].x = a[i] * as + b[i] * bs;
		out[i].y = 0;
	}

	// Real: out = a - b
	__kernel void sub(__global float *out, __global float *a, __global float *b) {
		const size_t i = get_global_id(0);
		out[i] = a[i] - b[i];
	}

	float soft_clamp(float x, float lo, float hi) {
		if(x < lo) return x - lo;
		if(x > hi) return x - hi;
		return 0;
	}

	// Real: out = [a + b * s]
	__kernel void mul_add_clamp(__global float2 *out, __global float2 *a, __global float2 *b, float s, float lo, float hi) {
		const size_t i = get_global_id(0);
		out[i].x = soft_clamp(a[i].x + b[i].x * s, lo, hi);
		out[i].y = 0;
	}
)";


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
	auto prog = ctx.compile(kernels);
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
		auto k = constraints[i]->get_k(x_in);
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
		debug_r(max_norm, "reduction");
		multi_array<float, 2> max_norm_a(size);
		seq = seq.then()(max_norm >> max_norm_a);
		seq.then().resume();
		max_norm_v[i] = max(max_norm_a);
		//seq = seq.then()( max_norm >> max_norm_v[i] );

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
	const float q = 1; // TODO

	// Repeat until good enough.
	for(size_t n = 0 ; n < max_steps ; n++) {
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
			auto lo = -q * sigma;
			auto hi = q * sigma;
			seq = seq.then()( mul_add_clamp(y[i], y[i], convolved, sigma, lo, hi) );
			debug_c(y[i], "new y");

			// convolve y_i with conjugate transpose of kernel
			seq = seq.then()( fft.forward(y[i], fft_conv) );
			debug_c(fft_conv, "y[i] forward fft");
			seq = seq.then()( complex_mul(fft_conv, fft_conv, fft_conj_k[i]) );
			debug_c(fft_conv, "kernel' * y[i]");
			seq = seq.then()( fft.backward(fft_conv, convolved) );
			debug_c(convolved, "backward fft");
			
			// accumulate
			seq = seq.then()( add2v.args(w, w, convolved) );
			debug_r(w, "accumulate");
		}

		// new x
		auto tau_ = 1 / (1 + tau), tau_tau = tau / (1 + tau), tau_tau_n = -tau_tau;
		seq = seq.then()( mul_add3vs(new_x, x, tau_, w, tau_tau_n, Y, tau_tau) );
		debug_r(new_x, "x - w + Y");

		// theta
		const float theta = 1 / sqrt(1 + 2 * tau * gamma);

		tau *= theta;
		sigma /= theta;

		// new bar_x
		auto theta1 = theta + 1, theta_n = -theta;
		seq = seq.then()( mul_add2vs(bar_x, new_x, theta1, x, theta_n) );
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
