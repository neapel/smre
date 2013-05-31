#include "chambolle_pock.h"
#include <boost/program_options.hpp>
#include "constraint_parser.h"
#include "watch.h"



using namespace std;
using namespace boost;
using namespace boost::program_options;

static bool run_cpu = false, run_gpu = false, profile = false;
static size_t runs = 10;


template<template<class> class impl, class T>
void run(params<T> a, multi_array<T, 2> in) {
	watch w;
	auto prof = make_shared<vex::profiler>(vex::current_context().queue());
	try {
		impl<T> c(a);
		if(profile) c.profiler = prof;
		c.run(in);
		for(size_t i = 0 ; i < runs ; i++) {
			w.start();
			c.run(in);
			w.stop();
		}
	} catch(...) {}
	if(profile) {
		cout << *prof << endl;
	} else {
		cout << '\t';
		if(w.times.size() > 0) cout << w.median();
		else cout << "nan";	
	}
}


template<class T>
void bench(size_t image, size_t kernels) {
	vector<size_t> kernel_sizes;
	size2_t sz{{image, image}};
	for(size_t i = 0 ; i < kernels ; i++)
		kernel_sizes.push_back(1);
	params<T> p(sz, kernel_sizes);
	p.force_q = 3;
	multi_array<T, 2> in(extents[image][image]);
	mimas::fill(in, 1);
	cout << image << '\t' << kernels;
	if(run_cpu) run<chambolle_pock_cpu>(p, in);
	if(run_gpu) {
		run<chambolle_pock_gpu>(p, in);
		p.use_fft = false;
		run<chambolle_pock_gpu>(p, in);
	}
	cout << endl;
}


int main(int argc, char **argv) {
	options_description desc("Options");
	sizes_t sizes{128}, kernels{4};
	desc.add_options()
		("help,h", "show help")
		("size,s", value(&sizes)->default_value(sizes), "list of sizes to try.")
		("kernels,k", value(&kernels)->default_value(kernels), "list of kernel counts to try.")
		("gpu", bool_switch(&run_gpu), "use gpu")
		("cpu", bool_switch(&run_cpu), "use cpu")
		("runs,r", value(&runs)->default_value(10), "number of runs to measure")
		("profile,p", bool_switch(&profile), "create profile instead of benchmark");
	variables_map vm;
	store(parse_command_line(argc, argv, desc), vm);
	notify(vm);
	if(vm.count("help") || !(run_gpu | run_cpu)) {
		cerr << desc << endl;
		return EXIT_FAILURE;
	}

	vex::Context clctx(vex::Filter::Count(1));
	vex::StaticContext<>::set(clctx);
	cerr << "CL context:" << clctx << endl;

	if(!profile) {
		cout << "size\tkernels";
		if(run_cpu) cout << "\tcpu";
		if(run_gpu) cout << "\tgpu\tgpusat";
		cout << endl;
	}

	for(auto w : sizes)
		for(auto k : kernels)
			bench<float>(w, k);

	return EXIT_SUCCESS;
}
