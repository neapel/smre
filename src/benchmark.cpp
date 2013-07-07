#include "chambolle_pock.h"
#include <boost/program_options.hpp>
#include "constraint_parser.h"

using namespace std;
using namespace boost;
using namespace boost::program_options;

static bool run_cpu = false, run_gpu = false, profile = false;
static size_t runs = 10;

typedef float T;


void run(params<T> p, multi_array<T,2> in) {
	vex::stopwatch<> w;
	auto prof = make_shared<vex::profiler<>>(vex::current_context().queue());
	try {
		auto c = p.runner();
		if(profile) c->profiler = prof;
		c->run(in);
		for(size_t i = 0 ; i < runs ; i++) {
			w.tic();
			in = c->run(in);
			w.toc();
		}
	} catch(...) {}
	if(profile) {
		cout << *prof << endl;
	} else {
		cout << '\t';
		if(w.tics() > 0) cout << w.average();
		else cout << "nan";	
	}
}


void bench(params<T> p, size_t image, size_t kernels) {
	size2_t sz{{image, image}};
	p.size = sz;
	p.kernel_sizes.clear();
	for(size_t i = 0 ; i < kernels ; i++)
		p.kernel_sizes.push_back(1);
	multi_array<T, 2> in(extents[image][image]);
	mimas::fill(in, 1);
	cout << image << '\t' << kernels;
	if(run_cpu) {
		p.use_gpu = false;
		p.use_fft = true;
		run(p, in);
	}
	if(run_gpu) {
		p.use_gpu = true;
		p.use_fft = true;
		run(p, in);
		p.use_fft = false;
		run(p, in);
	}
	cout << endl;
}


int main(int argc, char **argv) {
	params<T> base_p;
	base_p.force_q = 3;
	base_p.input_stddev = 1;
	base_p.tolerance = -1;
	base_p.max_steps = 100;

	options_description desc("Options");
	sizes_t sizes{128}, kernels{4};
	desc.add_options()
		("help,h", "show help")
		("size,s", value(&sizes)->default_value(sizes), "list of sizes to try.")
		("kernels,k", value(&kernels)->default_value(kernels), "list of kernel counts to try.")
		("resolvent", value(&base_p.resolvent)->default_value(base_p.resolvent),
				"Resolvent function to use, either “L2” for L₂ or “H1 <delta>” for H₁")
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
			bench(base_p, w, k);

	return EXIT_SUCCESS;
}
