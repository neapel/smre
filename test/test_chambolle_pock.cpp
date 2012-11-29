#include "chambolle_pock.h"
#include "gil_io.h"
#include <boost/program_options.hpp>
#include <boost/regex.hpp>
#include <stdexcept>

using namespace mimas;
using namespace std;
using namespace boost;
using namespace boost::program_options;

template<class A>
void normalize(A &in) {
	auto lo = min(in), hi = max(in);
	cerr << " lo=" << lo << " hi=" << hi << endl;
	if(lo > 0) lo = 0;
	if(hi < 1) hi = 1;
//	lo = -1; hi = 1;
	in -= lo;
	in /= hi - lo;
}

multi_array<float, 2> gauss(size_t w, size_t h, float sigma) {
	multi_array<float, 2> out(extents[h][w]);
	const float den = -1 / (2 * pow(sigma, 2));
	float x0 = w/2.0, y0 = h/2.0;
	for(size_t y = 0 ; y < h ; y++)
		for(size_t x = 0 ; x < w ; x++)
			out[y][x] = exp( den * (pow(x - x0, 2) + pow(y - y0, 2)) );
	return out;
}

struct constraint_options {
	float a, b;
	std::function<multi_array<float, 2>(size_t, size_t)> create;
};

void validate(any &v, const vector<string> &values, constraint_options *, int) {
	static regex r("(gauss:(?<gauss>\\d+\\.?\\d*)|box:(?<box>\\d+))(,(?<a>-?\\d*\\.?\\d*),(?<b>\\d*\\.?\\d*))?");
	validators::check_first_occurrence(v);
	const string &s = validators::get_single_string(values);
	smatch m;
	if(regex_match(s, m, r)) {
		const float a = m["a"].matched ? lexical_cast<float>(m["a"]) : -1;
		const float b = m["b"].matched ? lexical_cast<float>(m["b"]) :  1;
		if(m["gauss"].matched) {
			const auto sigma = lexical_cast<float>(m["gauss"]);
			v = any(constraint_options{a, b, [=](size_t w, size_t h) {
				auto k = gauss(w, h, sigma);
				k /= sum(k);
				return k;
			}});
		} else if(m["box"].matched) {
			const auto size = lexical_cast<size_t>(m["box"]);
			v = any(constraint_options{a, b, [=](size_t, size_t) {
				multi_array<float, 2> k(extents[size][size]);
				fill(k, 1 / sqrt(1.0 * size * size));
				return k;
			}});
		}
	} else {
		throw validation_error(validation_error::invalid_option_value);
	}
}


int main(int argc, char **argv) {

	string input_file;
	string output_prefix = "";
	float tau;
	vector<constraint_options> pre_constraints;

	{
		options_description desc("Options");
		desc.add_options()
			("help,h", "show this message")
			("input,i", value(&input_file)->required(), "input PNG image")
			("output,o", value(&output_prefix), "output folder")
			("tau", value(&tau)->required(), "initial value for tau")
			("constraints", value(&pre_constraints)->required(), "constraints: 'gauss:SIGMA[,A,B]' or 'box:SIZE[,A,B]'");

		positional_options_description pos;
		pos.add("constraints", -1);

		variables_map vm;
		try {
			store(command_line_parser(argc, argv)
				.options(desc).positional(pos).run(), vm);
			notify(vm);
		} catch(error &e) {
			cerr << "Error parsing command line: " << e.what() << '\n';
			cerr << desc << endl;
			return EXIT_FAILURE;
		}
		if(vm.count("help")) {
			cout << desc;
			return EXIT_SUCCESS;
		}
	}

	// read input
	auto x = read_image(input_file);
	const size_t w = x.shape()[0], h = x.shape()[1];
	x *= 2.0f;
	x -= 1.0f;

	// constraints and their norms
	vector<constraint> constraints;
	const auto size = extents_of(x);
	multi_array<complex<float>, 2> padded_kernel(size), kernel_fft(size);
	double constr_norm2 = 0;
	int i = 0;
	for(auto c : pre_constraints){
		constraints.push_back(constraint{c.a, c.b, x, c.create(w, h)});
		kernel_pad(constraints[i].k, padded_kernel);
		fftw::forward(padded_kernel, kernel_fft)();
		i += 1;
		double local_constr = 0;
		for(size_t iy = 0 ; iy < size[0] ; iy++)
			for(size_t ix = 0 ; ix < size[1] ; ix++)
				local_constr = max(pow(real(kernel_fft[ix][iy]),2) + pow(imag(kernel_fft[ix][iy]),2), local_constr);

		constr_norm2 += local_constr;
	}
	cerr << "Norm of constraints=" << constr_norm2 << endl;
	// run
	const float sigma = 1.0f / (tau * constr_norm2);
	const float gamma = 1.0f;

	chambolle_pock(tau, sigma, gamma, x, constraints, [=](const multi_array<float, 2> &x, string name){
		cerr << name;
		multi_array<float, 2> xn = x;
		normalize(xn);
		write_image(output_prefix + name + ".png", xn);

	});

	return EXIT_SUCCESS;
}
