#include "chambolle_pock.h"
#include "constraint_parser.h"
#include "watch.h"
#include <boost/program_options.hpp>


using namespace std;
using namespace boost;
using namespace boost::program_options;

static bool run_cpu = true, run_gpu = true;
static size_t runs = 10;


template<impl_t impl, class T>
void run(params<T> a, multi_array<T, 2> in) {
	vector<double> times;
	try {
		watch w_first;
		chambolle_pock<impl, T> c(a);
		c.run(in);
		times.push_back(w_first());
		if(runs > 0) {
			watch w_runs;
			for(size_t i = 0 ; i < runs ; i++) {
				watch w_run;
				c.run(in);
				times.push_back(w_run());
			}
			times.push_back(w_runs() / runs);
		}
	} catch(...) {}
	for(size_t i = 0 ; i < runs + 2 ; i++) {
		if(i < times.size())
			cout << '\t' << times[i];
		else
			cout << "\tnan";
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
	fill(in, 1);
	cout << image << '\t' << kernels;
	if(run_cpu) run<CPU_IMPL>(p, in);
	if(run_gpu) run<GPU_IMPL>(p, in);
	cout << endl;
}

struct sizes_t : vector<size_t> {
	sizes_t() : vector<size_t>() {}
	sizes_t(initializer_list<size_t> l) : vector<size_t>(l) {}
	sizes_t(vector<size_t> l) : vector<size_t>(l) {}
};

void validate(any &v, const vector<string> &values, sizes_t *, int) {
	//?? validators::check_first_occurence(v);
	auto str = validators::get_single_string(values);
	try {
		auto lst = list_expression(str);
		v = any(sizes_t(lst));
	} catch(...) {
		throw validation_error(validation_error::invalid_option_value);
	}
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
	if(run_cpu) {
		cout << "\tcpufirst";
		for(size_t i = 0 ; i < runs ; i++) cout << "\tcpu" << i;
		cout << "\tcpuavg";
	}
	if(run_gpu) {
		cout << "\tgpufirst";
		for(size_t i = 0 ; i < runs ; i++) cout << "\tgpu" << i;
		cout << "\tgpuavg";
	}
	cout << endl;

	for(auto w : sizes)
		for(auto k : kernels)
			bench<float>(w, k);

	return EXIT_SUCCESS;
}
