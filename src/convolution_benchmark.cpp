#include <iostream>
#include <boost/program_options.hpp>
#include "constraint_parser.h"
#include "convolution.h"
#include "watch.h"
#include "multi_array_operators.h"

using namespace std;
using namespace boost;
using namespace boost::program_options;

typedef float T;
typedef multi_array<T, 2> A;

size_t kernel_runs = 10, image_runs = 10;

vex::Context ctx(vex::Filter::Count(1));


template<class Conv>
void conv(size_t sz, const typename Conv::A &x, typename Conv::A &y, size_t h) {
	auto c = new Conv(size2_t{{sz, sz}});

	double t_prep, t_img, t_conv, t_total;

	watch w_total;
	watch w_prep;
	auto k = c->prepare_kernel(h, false);
	ctx.queue()[0].finish();
	t_prep = w_prep();

	for(size_t ir = 0 ; ir < image_runs ; ir++) {
		watch w_img;
		auto i = c->prepare_image(x);
		ctx.queue()[0].finish();
		t_img += w_img();

		watch w_conv;
		for(size_t kr = 0 ; kr < kernel_runs ; kr++)
			c->conv(i, k, y);
		ctx.queue()[0].finish();
		t_conv += w_conv() / kernel_runs;
	}

	t_img /= image_runs;
	t_conv /= image_runs;
	t_total = w_total();

	cout << '\t' << t_prep << '\t' << t_img << '\t' << t_conv << '\t' << t_total;

	delete c;
}

template<class Conv>
void bench_cpu(size_t sz, size_t hs) {
	multi_array<T, 2> x(extents[sz][sz]), y(x);
	fill(x, 1);
	conv<Conv>(sz, x, y, hs);
}

template<class Conv>
void bench_gpu(size_t sz, size_t hs) {
	vex::vector<T> x(sz * sz), y(sz * sz);
	x = 1;
	conv<Conv>(sz, x, y, hs);
}


int main(int argc, char **argv) {
	options_description desc("Options");
	sizes_t sizes{{128}}, hs{{4}};
	bool run_gpu = true, run_cpu = true, run_fft = true, run_sat = true;
	desc.add_options()
		("help", "show help")
		("size", value(&sizes), "list of sizes to try.")
		("box", value(&hs), "list of box sizes to try.")
		("gpu", value(&run_gpu), "use gpu")
		("cpu", value(&run_cpu), "use cpu")
		("fft", value(&run_fft), "use fft")
		("sat", value(&run_sat), "use sat")
		("kernel-runs", value(&kernel_runs), "run each kernel times")
		("image-runs", value(&kernel_runs), "run each image times");
	variables_map vm;
	store(parse_command_line(argc, argv, desc), vm);
	notify(vm);
	if(vm.count("help")) {
		cerr << desc << endl;
		return EXIT_FAILURE;
	}

	vex::StaticContext<>::set(ctx);

	cout << "size\tbox";
	if(run_gpu && run_fft) cout << "\tgpufftkprep\tgpufftiprep\tgpufftconv\tgpuffttotal";
	if(run_gpu && run_sat) cout << "\tgpusatkprep\tgpusatiprep\tgpusatconv\tgpusattotal";
	if(run_cpu && run_fft) cout << "\tcpufftkprep\tcpufftiprep\tcpufftconv\tcpuffttotal";
	if(run_cpu && run_sat) cout << "\tcpusatkprep\tcpusatiprep\tcpusatconv\tcpusattotal";
	cout << endl;

	for(auto sz : sizes)
		for(auto h : hs) {
			cout << sz << '\t' << h;
			if(run_gpu && run_fft) bench_gpu<gpu_fft_convolver<T>>(sz, h);
			if(run_gpu && run_sat) bench_gpu<gpu_sat_convolver<T>>(sz, h);
			if(run_cpu && run_fft) bench_cpu<cpu_fft_convolver<T>>(sz, h);
			if(run_cpu && run_sat) bench_cpu<cpu_sat_convolver<T>>(sz, h);
			cout << endl;
		}

	return EXIT_SUCCESS;
	
}


