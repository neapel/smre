#include "chambolle_pock.h"
#include "kernel_generator.h"
#include <fstream>

//#define BOOST_FILESYSTEM_NO_DEPRECATED 1
#include <boost/filesystem.hpp>

using namespace boost;
using namespace boost::filesystem;
using namespace std;

multi_array<float, 2> constraint::get_k(const multi_array<float, 2> &img) {
	auto h = img.shape()[0], w = img.shape()[1];
	return kernel_from_string(expr)(w, h);
}


static const auto cache_dir = "cache/";
static const size_t quantils = 100;

float chambolle_pock::cached_q(std::array<size_t, 2> size, function<vector<float>()> calc) {
	// filename from kernel stack
	vector<string> kernels;
	for(auto c : constraints) kernels.push_back(c.expr);
	sort(kernels.begin(), kernels.end());
	ostringstream ss; ss << cache_dir << size[0] << 'x' << size[1];
	for(auto k : kernels) ss << ' ' << k;

	auto raw_target = ss.str() + ".simulation";
	auto quant_target = ss.str() + ".quantiles";
	create_directories(cache_dir);

	vector<float> qs;
	if(no_cache || !exists(quant_target)) {
		// simulate
		qs = calc();
		sort(qs.begin(), qs.end());
		// write raw output.
		ofstream fr(raw_target, ofstream::binary);
		for(const float &x : qs) fr.write(reinterpret_cast<const char *>(&x), sizeof(float));
		// write quantiles human-readable.
		ofstream fq(quant_target, ofstream::binary);
		for(size_t i = 0 ; i <= quantils ; i++)
			fq.write(reinterpret_cast<const char *>(
				&qs[size_t(float(qs.size() - 1) * i / quantils)]), sizeof(float));
	} else {
		// read
		ifstream f(quant_target, ifstream::binary);
		while(true) {
			float value;
			f.read(reinterpret_cast<char *>(&value), sizeof(float));
			if(!f) break;
			qs.push_back(value);
		}
	}

	// return (1 - alpha) quantile:
	return qs[size_t((qs.size() - 1) * (1 - alpha))];
}
