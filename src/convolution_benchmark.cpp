#include <iostream>
#include <boost/program_options.hpp>
#include "constraint_parser.h"
#include "convolution.h"
#include "multi_array_operators.h"

using namespace std;
using namespace boost;
using namespace boost::program_options;

typedef float T;
typedef multi_array<T, 2> A;

size_t runs = 10;

vex::Context ctx(vex::Filter::Count(1));


template<class Conv>
void conv(size_t sz, const typename Conv::A &x, typename Conv::A &y, size_t h) {
	auto c = make_shared<Conv>(size2_t{{sz, sz}});
	vex::stopwatch<> w_prep, w_img, w_conv, w_total;
	ctx.queue()[0].finish();
	for(size_t ir = 0 ; ir < runs ; ir++) {
		w_total.tic();

		w_prep.tic();
		auto k = c->prepare_kernel(h, false);
		ctx.queue()[0].finish();
		w_prep.toc();

		w_img.tic();
		auto i = c->prepare_image(x);
		ctx.queue()[0].finish();
		w_img.toc();

		w_conv.tic();
		c->conv(i, k, y);
		ctx.queue()[0].finish();
		w_conv.toc();

		w_total.toc();
	}
	const size_t s = sz * sz;
	cout << '\t' << (s / w_prep.average())
	     << '\t' << (s / w_img.average())
		  << '\t' << (s / w_conv.average())
		  << '\t' << (s / w_total.average());
}

template<class Conv>
void bench_cpu(size_t sz, size_t hs) {
	multi_array<T, 2> x(extents[sz][sz]), y(x);
	mimas::fill(x, 1);
	conv<Conv>(sz, x, y, hs);
}

template<class Conv>
void bench_gpu(size_t sz, size_t hs) {
	try {
		vex::vector<T> x(sz * sz), y(sz * sz);
		x = 1;
		conv<Conv>(sz, x, y, hs);
	} catch(...){
		cout << "\tnan\tnan\tnan\tnan";
	}
}

void bench_gpu_lin(size_t sz) {
	try {
		vex::vector<T> x(sz * sz);
		x = 1;
		x += 1;
		ctx.queue()[0].finish();
		vex::stopwatch<> w;
		for(size_t i = 0 ; i < runs ; i++) {
			ctx.queue()[0].finish();
			w.tic();
			x += 1;
			ctx.queue()[0].finish();
			w.toc();
		}
		cout << '\t' << (sz * sz / w.average());
	} catch(...) {
		cout << "\tnan";
	}
}


int main(int argc, char **argv) {
	options_description desc("Options");
	sizes_t sizes{128}, hs{4};
	bool run_gpu = true, run_cpu = true, run_fft = true, run_sat = true;
	desc.add_options()
		("help", "show help")
		("size", value(&sizes), "list of sizes to try.")
		("box", value(&hs), "list of box sizes to try.")
		("gpu", value(&run_gpu), "use gpu")
		("cpu", value(&run_cpu), "use cpu")
		("fft", value(&run_fft), "use fft")
		("sat", value(&run_sat), "use sat")
		("runs", value(&runs), "run multiple times");
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
	if(run_gpu           ) cout << "\tgpulin";
	cout << endl;

	for(auto sz : sizes) {
		for(auto h : hs) {
			cout << sz << '\t' << h;
			if(run_gpu && run_fft) bench_gpu<gpu_fft_convolver<T>>(sz, h);
			if(run_gpu && run_sat) bench_gpu<gpu_sat_convolver<T>>(sz, h);
			if(run_cpu && run_fft) bench_cpu<cpu_fft_convolver<T>>(sz, h);
			if(run_gpu           ) bench_gpu_lin(sz);
			cout << endl;
		}
	}

	return EXIT_SUCCESS;
	
}


