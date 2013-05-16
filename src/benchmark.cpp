#include "chambolle_pock.h"
#include <boost/program_options.hpp>
#include "constraint_parser.h"
#include "watch.h"



using namespace std;
using namespace boost;
using namespace boost::program_options;

static bool run_cpu = true, run_gpu = true;
static size_t runs = 10;


template<impl_t impl, class T>
void run(params<T> a, multi_array<T, 2> in) {
	watch w;
	try {
		chambolle_pock<impl, T> c(a);
		c.run(in);
		for(size_t i = 0 ; i < runs ; i++) {
			w.start();
			c.run(in);
			w.stop();
		}
	} catch(...) {}
	cout << '\t';
	if(w.times.size() > 0) cout << w.median();
	else cout << "nan";
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
	if(run_cpu) run<CPU_IMPL>(p, in);
	if(run_gpu) {
		run<GPU_IMPL>(p, in);
		p.use_fft = false;
		run<GPU_IMPL>(p, in);
	}
	cout << endl;
}


int main(int argc, char **argv) {
	options_description desc("Options");
	sizes_t sizes{{128}}, kernels{{4}};
	desc.add_options()
		("help", "show help")
		("size", value(&sizes), "list of sizes to try.")
		("kernels", value(&kernels), "list of kernel counts to try.")
		("gpu", value(&run_gpu), "use gpu")
		("cpu", value(&run_cpu), "use cpu")
		("runs", value(&runs), "number of runs to measure");
	variables_map vm;
	store(parse_command_line(argc, argv, desc), vm);
	notify(vm);
	if(vm.count("help")) {
		cerr << desc << endl;
		return EXIT_FAILURE;
	}

	vex::Context clctx(vex::Filter::Count(1));
	vex::StaticContext<>::set(clctx);
	cerr << "CL context:" << clctx << endl;

	cout << "size\tkernels";
	if(run_cpu) cout << "\tcpu";
	if(run_gpu) cout << "\tgpu\tgpusat";
	cout << endl;

	for(auto w : sizes)
		for(auto k : kernels)
			bench<float>(w, k);

	return EXIT_SUCCESS;
}
