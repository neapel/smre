#include <iostream>
#include <complex>

#include "opencl_multi_array.h"
#include "multi_array.h"
#include "multi_array_operators.h"
#include "multi_array_fft.h"
#include "gil_io.h"

using namespace cl;
using namespace boost;
using namespace std;
using namespace mimas;

typedef complex<float> cfloat;
typedef multi_array<float, 2> float_a;
typedef multi_array<cfloat, 2> cfloat_a;


cfloat_a to_complex(float_a in) {
	auto out = empty_clone<cfloat>(in);
	for(size_t y = 0 ; y < in.shape()[1] ; y++)
		for(size_t x = 0 ; x < in.shape()[0] ; x++)
			out[y][x] = in[y][x];
	return out;
}

float_a from_complex(cfloat_a in) {
	auto out = empty_clone<float>(in);
	for(size_t y = 0 ; y < in.shape()[1] ; y++)
		for(size_t x = 0 ; x < in.shape()[0] ; x++)
			out[y][x] = in[y][x].real();
	return out;
}

template<class T>
multi_array<T, 2> pad(const multi_array<T, 2> &in, const size_t *shape) {
	multi_array<T, 2> out(extents[shape[1]][shape[0]]);
	kernel_pad(in, out);
	return out;
}


// the different convolutions:

/** Naive convolution of real data. */
float_a convolution_cpu_naive(const float_a &in, const float_a &kernel) {
	const size_t w = in.shape()[0], h = in.shape()[1],
	            kw = kernel.shape()[0], kh = kernel.shape()[1];
	auto out = empty_clone<float>(in);
	const size_t dx = w - kw/2, dy = h - kh/2; // kernel origin = center
	for(size_t i = 0 ; i < h ; i++)
		for(size_t j = 0 ; j < w ; j++) {
			float sum = 0;
			for(size_t ki = 0 ; ki < kh ; ki++)
				for(size_t kj = 0 ; kj < kw ; kj++)
					sum += in[(i+ki+dy) % w][(j+kj+dx) % h] * kernel[ki][kj];
			out[i][j] = sum;
		}

	return out;
}


/** Convolve by computing R2C FFTs of data and kernel, return C2R FFT of product. */
float_a convolution_cpu_fft(const float_a &in, const float_a &kernel_small) {
	const auto kernel = pad(kernel_small, in.shape());
	const size_t h = in.shape()[1], w = in.shape()[0];
	cfloat_a in_f(extents[h][w/2+1]), kernel_f(extents[h][w/2+1]);
	fftw::forward(in, in_f)();
	fftw::forward(kernel, kernel_f)();
	auto out_c = in_f * kernel_f;
	auto out = empty_clone<float>(in);
	fftw::backward(out_c, out)();
	out /= float(w * h);
	return out;
}

#if HAVE_OPENCL
/** Naive convolution of real data on the CL device. */
float_a convolution_cl_naive(context &ctx, const float_a &in, const float_a &kernel) {
	const unsigned int h = in.shape()[1], w = in.shape()[0], kh = kernel.shape()[1], kw = kernel.shape()[0];
	auto out = empty_clone<float>(in);
	buffer b_in(sizeof(float) * w * h, stream_in),
			b_out(sizeof(float) * w * h, stream_out),
			b_kernel(sizeof(float) * kw * kh, stream_in);
	
	auto fold = ctx.compile(R"(
		__kernel void fold(__global float *out, __global float *in, __global float *kern, const uint w, const uint h, const uint kw, const uint kh) {
			const uint x = get_global_id(0), y = get_global_id(1);
			const uint dx = w - kw/2, dy = h - kh/2;
			float sum = 0;
			for(uint ky = 0 ; ky < kh ; ky++)
				for(uint kx = 0 ; kx < kw ; kx++)
					sum += in[((x+kx+dx) % w) + ((y+ky+dy) % h) * w] * kern[kx + ky * kw];
			out[x + y * w] = sum;
		}
	)")["fold"];

	after{
		ctx(b_in << in),
		ctx(b_kernel << kernel)
	}(fold.args(b_out, b_in, b_kernel, w, h, kw, kh).size({w, h}))
		.then()(b_out >> out)
		.then().resume();
	return out;
}

#if HAVE_AMD_FFT
/** Convolve by computing (on the CL device) R2C FFTs of data and kernel, return C2R FFT of product. */
float_a convolution_cl_fft(context &ctx, const float_a &in, const float_a &kernel_small) {
	auto kernel = pad(to_complex(kernel_small), in.shape());
	auto in_c = to_complex(in);
	auto out_c = empty_clone<cfloat>(in);
	const unsigned int h = in.shape()[1], w = in.shape()[0];
	const unsigned int N = sizeof(complex<float>) * w * h;
	buffer b_in(N, stream_in), b_temp(N, temp), b_out(N, stream_out),
			 b_kernel_in(N, stream_in), b_kernel_out(N, temp);

	fft plan(w, h);
	auto mult = ctx.compile(R"(
		__kernel void multiply(__global float2 *data, __global float2 *kern) {
			const uint tid = get_global_id(0);
			const float dr = data[tid].x, di = data[tid].y, kr = kern[tid].x, ki = kern[tid].y;
			data[tid].x = dr * kr - di * ki;
			data[tid].y = dr * ki + di * kr;
		}
	)")["multiply"];

	after{
		ctx(b_in << in_c)
			.then()(plan.forward(b_in, b_temp)),
		ctx(b_kernel_in << kernel)
			.then()(plan.forward(b_kernel_in, b_kernel_out))
	}(mult.args(b_temp, b_kernel_out).size({w*h}))
		.then()(plan.backward(b_temp, b_out))
		.then()(b_out >> out_c)
		.then().resume();

	return from_complex(out_c);
}
#endif
#endif




int main(int argc, char **argv) {
	if(argc != 4) {
		cerr << "usage: " << argv[0] << " <input.png> <kernel.png> <output-prefix>" << endl;
		return EXIT_FAILURE;
	}
	string base(argv[3]);

	// load image
	auto in = read_image(argv[1]);

	// load and normalize kernel
#if 1
	auto kernel = read_image(argv[2]);
	float kernel_sum = 0;
	for(size_t i = 0 ; i < kernel.shape()[1] ; i++)
		for(size_t j = 0 ; j < kernel.shape()[0] ; j++)
			kernel_sum += kernel[i][j];
	kernel /= kernel_sum;
#else
	float_a kernel(extents[1][1]);
	kernel[0][0] = 1;
#endif

	// output
	auto out = empty_clone<float>(in);

	// run each method:
	cerr << "cpu fft" << endl;
	write_image(base + "-cpu-fft.png", convolution_cpu_fft(in, kernel));

#if 0
	cerr << "cpu naive" << endl;
	write_image(base + "-cpu-naive.png", convolution_cpu_naive(in, kernel));
#endif

#if HAVE_OPENCL
	context ctx;
	cerr << "opencl naive" << endl;
	write_image(base + "-cl-naive.png", convolution_cl_naive(ctx, in, kernel));

#if HAVE_AMD_FFT
	cerr << "opencl fft" << endl;
	write_image(base + "-cl-fft.png", convolution_cl_fft(ctx, in, kernel));
#endif
#endif

}
