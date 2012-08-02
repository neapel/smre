#include "opencl_multi_array.h"
#include "multi_array_operators.h"

#include <iterator>
#include <cassert>
#include <cmath>

using namespace cl;
using namespace std;
using namespace boost;
using namespace mimas;

int main(int, char**) {

	// make up some data
	const size_t n = 8;
	multi_array<float, 2> data_a(extents[n][n]), data_b(extents[n][n]), data_out(extents[n][n]);
	iota(data_a, 0.0f);
	iota(data_b, 0.0f);

	// create CL context
	context ctx;

	// create buffers
	const size_t bs = sizeof(float) * pow(n, data_a.num_dimensions());
	buffer input_a(bs, stream_in), input_b(bs, stream_in), output(bs, stream_out);

	// load kernel
	auto prog = ctx.compile(R"(
		kernel void add(
			global float *input_a,
			global float *input_b,
			global float *output
		) {
			const uint x = get_global_id(0);
			const uint y = get_global_id(1);
			const uint i = x + y * get_global_size(0);
			output[i] = input_a[i] + input_b[i];
		}
	)");

	// queue input, calculation, output, wait.
	after{
		after{
			after{
				ctx( input_a << data_a ),
				ctx( input_b << data_b )
			}( prog["add"].args(input_a, input_b, output).size({n, n}) )
		}( output >> data_out )
	}.resume();

	// verify data
	assert( data_out == data_a + data_b );
}

