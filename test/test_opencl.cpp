#include "opencl_helpers.h"

#include <iterator>
#include <cassert>
#include <cmath>

using namespace cl;
using namespace std;

int main(int, char**) {

	// make up some data
	const size_t n = 16;
	array<float, n> data_a, data_b;
	for(size_t i = 0 ; i < n ; i++) {
		data_a[i] = 1.0 * (i + 1);
		data_b[i] = 1.0 * (i + 1);
	}

	// create CL context
	context ctx;
	cerr << ctx << endl;

	// create buffers
	buffer input_a(stream_in),
	       input_b(sizeof(float) * n, stream_in),
	       output(sizeof(float) * n, stream_out);

	// load kernel
	auto prog = ctx.compile(R"(
		__kernel void f(
			__global float *input_a,
			const float mul_a,
			__global float *input_b,
			const float mul_b,
			__global float *output
		) {
			uint tid = get_global_id(0);
			output[tid] = input_a[tid] * mul_a + input_b[tid] * mul_b;
		}
	)");
	cerr << prog << endl;

	auto multiply_add = prog["f"];
	cerr << multiply_add << endl;

	// queue fill buffers
	auto a_written = ctx( input_a << data_a );
	auto b_written = ctx( input_b << data_b );

	// queue run kernel
	const float mul_a = 2.0f, mul_b = 0.5f;
	auto add_done = after{a_written, b_written}(
		multiply_add.args(input_a, mul_a, input_b, mul_b, output).size({n})
	);

	// queue retrieve data
	array<float, n> data_out;
	auto o_read = after{add_done}( output >> data_out );

	// wait until ready
	after{o_read}.resume();

	// verify data
	for(size_t i = 0 ; i < n ; i++) {
		const float expect = data_a[i] * mul_a + data_b[i] * mul_b;
		assert( abs(data_out[i] - expect) < 0.001 );
	}
}
