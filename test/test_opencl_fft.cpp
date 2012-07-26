#include <iostream>
#include <complex>

std::ostream &operator<<(std::ostream &o, const std::complex<float> &c) {
	o << c.real();
	if(c.imag() != 0) {
		if(c.imag() > 0) o << '+';
		o << c.imag() << 'i';
	}
	return o;
}

#include "opencl_multi_array.h"
#include "multi_array_operators.h"
#include "multi_array_print.h"
#include "multi_array_fft.h"

using namespace cl;
using namespace boost;
using namespace std;
using namespace mimas;

const size_t w = 16, h = 8, kw = 4, kh = 4;
typedef multi_array<complex<float>, 2> A;

void convolution_cpu_naive(const A &in, const A &kernel, A &out) {
	for(size_t i = 0 ; i < w ; i++)
		for(size_t j = 0 ; j < h ; j++) {
			complex<float> sum = 0;
			for(size_t ki = 0 ; ki < kw ; ki++)
				for(size_t kj = 0 ; kj < kw ; kj++) {
					sum += in[(i+ki) % w][(j+kj) % h] * kernel[ki][kj];
				}
			out[i][j] = sum;
		}
}

void convolution_cl_naive(const A &in, const A &kernel, A &out) {
	(void)in; (void)kernel; (void)out;
}

void convolution_cpu_fft(const A &in, const A &kernel, A &out) {
	A in_(extents[w][h]), kernel_(extents[w][h]);
	fftw::forward(in, in_)();
	fftw::forward(kernel, kernel_)();
	for(size_t i = 0 ; i < w ; i++)
		for(size_t j = 0 ; j < h ; j++)
			out[i][j] = in_[i][j] * kernel_[i][j];
	fftw::backward(out, out)();
	for(size_t i = 0 ; i < w ; i++)
		for(size_t j = 0 ; j < h ; j++)
			out[i][j] /= w * h;
}

void convolution_cl_fft(context &ctx, const A &in, const A &kernel, A &out) {
	const size_t N = sizeof(complex<float>) * w * h;
	buffer b_in(N, stream_in), b_temp(N, temp), b_out(N, stream_out),
			 b_kernel_in(N, stream_in), b_kernel_out(N, full_access);

	fft plan(w, h);
	auto mult = ctx.compile(R"(
		kernel void multiply(global float2 *data, constant float2 *kern) {
			const uint tid = get_global_id(0);
			const float dr = data[tid].x, di = data[tid].y, kr = kern[tid].x, ki = kern[tid].y;
			data[tid].x = dr * kr - di * ki;
			data[tid].y = di * kr - dr * ki;
		}
	)")["multiply"];

	after{
		ctx(b_in << in)
			.then()(plan.forward(b_in, b_temp)),
		ctx(b_kernel_in << kernel)
			.then()(plan.forward(b_kernel_in, b_kernel_out))
	}(mult.args(b_temp, b_kernel_out).size({N/2}))
		.then()(plan.backward(b_temp, b_out))
		.then()(b_out >> out)
		.then().resume();
}


int main(int, char**) {
	context ctx;

	// arrays
	A in(extents[w][h]), out(extents[w][h]), kernel(extents[w][h]);

	for(size_t i = 0 ; i < w ; i++)
		for(size_t j = 0 ; j < h ; j++)
			in[i][j] = 1.0f * rand() / RAND_MAX;
	for(size_t i = 0 ; i < kw ; i++)
		for(size_t j = 0 ; j < kh ; j++)
			kernel[i][j] =  1.0f * rand() / RAND_MAX;

	cout << in << " in\n" << kernel << " kernel\n";

	convolution_cpu_fft(in, kernel, out);
	cout << out << " cpu fft" << endl;

	convolution_cpu_naive(in, kernel, out);
	cout << out << " cpu naive" << endl;

	//convolution_cl_fft(ctx, in, kernel, out);
	//cout << out << " cl fft" << endl;

}
