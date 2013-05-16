/** Check numerical error of convolutions against arbitrary-precision SAT */

#include <boost/multiprecision/mpfr.hpp>
#include <boost/program_options.hpp>
#include "constraint_parser.h"
#include "convolution.h"
using namespace boost;
using namespace boost::multiprecision;

typedef boost::multiprecision::static_mpfr_float_50 float50;
//typedef boost::multiprecision::mpfr_float_1000 float50;

template<class C>
void check(std::string id, std::string name, size2_t size, size_t runs, sizes_t hs) {
	multi_array<float50,2> elem_diff(size), elem_ref(size), in(size), ref(size);
	multi_array<float,2> in_(size), out_(size);
	std::uniform_real_distribution<float> dist(-0.5, 0.5);

	mimas::fill(elem_diff, float50(0));
	mimas::fill(elem_ref, float50(0));

	C c_f(size);
	cpu_sat_convolver<float50> c_r(size);

	for(size_t h : hs) {
		auto k_f = c_f.prepare_kernel(h, false);
		auto k_r = c_r.prepare_kernel(h, false);
		std::vector<float50> errs;

		for(size_t run = 0 ; run < runs ; run++) {
			std::minstd_rand gen(23 + run);
			for(auto r : in_) for(auto &v : r) v = dist(gen);
			in = in_;

			auto i_f = c_f._prepare_image(in_);
			c_f._conv(i_f, k_f, out_);

			auto i_r = c_r._prepare_image(in);
			c_r._conv(i_r, k_r, ref);

			float50 total_diff(0), total_ref(0);
			for(size_t i0 = 0 ; i0 < size[0] ; i0++ ) {
				for(size_t i1 = 0 ; i1 < size[1] ; i1++) {
					float50 r(ref[i0][i1]), o(out_[i0][i1]), d = abs(o - r), d2 = d * d, r2 = r * r;
					total_diff += d2;
					elem_diff[i0][i1] += d2;
					total_ref += r2;
					elem_ref[i0][i1] += r2;
				}
			}
			errs.push_back(sqrt(total_diff / total_ref));
		}
		std::sort(errs.begin(), errs.end());
		std::cout << size[0] << '\t' << h;
		for(auto e : errs) std::cout << '\t' << e;
		std::cout << std::endl;
	}

	if(name != "-") {
		std::ofstream o(name + ".dat");
		for(size_t i0 = 0 ; i0 < size[0] ; i0++ ) {
			for(size_t i1 = 0 ; i1 < size[1] ; i1++)
				o << sqrt(elem_diff[i0][i1] / elem_ref[i0][i1]) << '\t';
			o << '\n';
		}
	}
	std::cout << std::endl;
}

int main(int argc, char **argv) {
	using namespace boost::program_options;
	size_t s, runs;
	sizes_t hs;
	std::string sat_gpu_f, sat_cpu_f, fft_gpu_f, fft_cpu_f;
	options_description desc("Options");
	desc.add_options()
		("help", "show help")
		("size", value(&s)->default_value(128), "image size")
		("box", value(&hs)->default_value(sizes_t{9}), "box size")
		("sat-gpu", value(&sat_gpu_f)->default_value("")->implicit_value("-"), "output filename")
		("sat-cpu", value(&sat_cpu_f)->default_value("")->implicit_value("-"), "output filename")
		("fft-gpu", value(&fft_gpu_f)->default_value("")->implicit_value("-"), "output filename")
		("fft-cpu", value(&fft_cpu_f)->default_value("")->implicit_value("-"), "output filename")
		("runs", value(&runs)->default_value(10), "number of runs to measure");
	variables_map vm;
	store(parse_command_line(argc, argv, desc), vm);
	notify(vm);
	if(vm.count("help")) {
		std::cerr << desc << std::endl;
		return EXIT_FAILURE;
	}

	vex::Context ctx(vex::Filter::Count(1));
	vex::StaticContext<>::set(ctx);
	size2_t size{{s,s}};

	std::cout << "size\th";
	for(size_t i = 0 ; i < runs ; i++) {
		std::cout << "\terr";
		if(runs > 1) std::cout << (100 * i / (runs-1));
	}
	std::cout << std::endl;

	if(sat_gpu_f.size() > 0) check<gpu_sat_convolver<float>>("satgpu", sat_gpu_f, size, runs, hs);
	if(sat_cpu_f.size() > 0) check<cpu_sat_convolver<float>>("satcpu", sat_cpu_f, size, runs, hs);
	if(fft_gpu_f.size() > 0) check<gpu_fft_convolver<float>>("fftgpu", fft_gpu_f, size, runs, hs);
	if(fft_cpu_f.size() > 0) check<cpu_fft_convolver<float>>("fftcpu", fft_cpu_f, size, runs, hs);

	return EXIT_SUCCESS;
}
