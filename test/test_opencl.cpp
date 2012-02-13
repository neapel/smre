#include "opencl_helpers.h"

#include <iterator>

using namespace cl;
using namespace std;

int main(int, char**) {
	
	context ctx;

	const size_t n = 256;
	array<float, n> data_a, data_b;
	for(size_t i = 0 ; i < n ; i++) {
		data_a[i] = 1.0 * i / n;
		data_b[i] = 1.0 * n / i;
	}

	auto input_a = ctx.create_buffer(data_a);
	auto input_b = ctx.create_buffer<float>(n);
	auto output = ctx.create_buffer<float>(n);

	auto b_written = input_b.write(data_b, 0, {});

	auto prog = ctx.load_program("test_opencl_add.cl");
	auto kern = prog["multiply_add"];
	kern.args(input_a, 2.0f, input_b, 1000.0f, output);

	auto ran = kern.run({n}, {}, {}, {b_written});

	array<float, n> data_out;
	auto o_read = output.read(data_out, 0, {ran});

	ctx.wait({o_read});

	cout << "output: ";
	copy(data_out.begin(), data_out.end(), ostream_iterator<float>(cout, ", "));
	cout << endl;

}
