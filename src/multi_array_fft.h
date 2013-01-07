#ifndef __MULTI_ARRAY_FFT_H__
#define __MULTI_ARRAY_FFT_H__

extern "C" {
	#include <fftw3.h>
}
#include <boost/multi_array.hpp>
#include <array>
#include <stdexcept>
#include <complex>

namespace fftw {

template<class In, class Out>
std::array<int, In::dimensionality> get_shape(const In &in, const Out &out) {
	const size_t N = In::dimensionality;
	static_assert(N == Out::dimensionality, "Arrays must have same rank.");
	if(!is_continuous(in) || !is_continuous(out))
		throw std::invalid_argument("Only continuous arrays are supported.");
	for(size_t i = 0 ; i < N ; i++)
		if(in.index_bases()[i] != 0 || out.index_bases()[i] != 0)
			throw std::invalid_argument("Nonzero bases are not supported.");
	std::array<int, N> shape;
	for(size_t i = 0 ; i < N ; i++)
		shape[i] = static_cast<int>(std::max(in.shape()[i], out.shape()[i]));
	return shape;
}

struct plan {
	fftwf_plan p;

	plan() : p(NULL) {}

	// pass ownership
	plan(plan &other) : p(other.p) { other.p = NULL; }
	plan(plan &&other) : p(other.p) { other.p = NULL; }
	plan &operator=(plan &other) { p = other.p; other.p = NULL; return *this; }
	plan &operator=(plan &&other) { p = other.p; other.p = NULL; return *this; }

	/** Complex to Complex FFT. */
	template<size_t dims>
	plan(const boost::multi_array<std::complex<float>, dims> &in, boost::multi_array<std::complex<float>, dims> &out, int dir, unsigned int flags) {
		switch(dir) {
			case FFTW_FORWARD: break;
			case FFTW_BACKWARD: break;
			default: throw std::invalid_argument("Only C2C Fourier transform.");
		}
		auto shape = get_shape(in, out);
		for(size_t i = 0 ; i < dims ; i++)
			if(in.shape()[i] != out.shape()[i])
				throw std::invalid_argument("Input and output arrays must be of same size.");

		p = fftwf_plan_dft(dims, shape.data(),
			const_cast<fftwf_complex *>(reinterpret_cast<const fftwf_complex *>(in.origin())),
			reinterpret_cast<fftwf_complex *>(out.origin()), dir, flags);
	}

	/** Real to Complex FFT. */
	template<size_t dims>
	plan(const boost::multi_array<float, dims> &in, boost::multi_array<std::complex<float>, dims> &out, int dir, unsigned int flags) {
		if(dir != FFTW_FORWARD)
			throw std::invalid_argument("R2C transform must be forward.");
		auto shape = get_shape(in, out);
		if(out.shape()[dims-1] != in.shape()[dims-1] / 2 + 1)
			throw std::invalid_argument("Width of output must be input width/2 + 1");
		for(size_t i = 0 ; i < dims-1 ; i++)
			if(in.shape()[i] != out.shape()[i])
				throw std::invalid_argument("Other dimensions must be same size.");

		p = fftwf_plan_dft_r2c(dims, shape.data(), const_cast<float *>(in.origin()),
				reinterpret_cast<fftwf_complex *>(out.origin()), flags);
	}

	/** Complex to Real FFT. */
	template<size_t dims>
	plan(const boost::multi_array<std::complex<float>, dims> &in, boost::multi_array<float, dims> &out, int dir, unsigned int flags) {
		if(dir != FFTW_BACKWARD)
			throw std::invalid_argument("C2R transform must be backward.");
		auto shape = get_shape(in, out);
		if(in.shape()[dims-1] != out.shape()[dims-1] / 2 + 1)
			throw std::invalid_argument("Width of input must be output width/2 + 1");
		for(size_t i = 0 ; i < dims-1 ; i++)
			if(in.shape()[i] != out.shape()[i])
				throw std::invalid_argument("Other dimensions must be same size.");

		p = fftwf_plan_dft_c2r(dims, shape.data(),
				const_cast<fftwf_complex *>(reinterpret_cast<const fftwf_complex *>(in.origin())),
				out.origin(), flags);
	}

	void operator()() {
		fftwf_execute(p);
	}

	~plan() {
		if(p != NULL)
			fftwf_destroy_plan(p);
	}
};






template<class In, class Out>
plan forward(const In &in, Out &out, unsigned int flags = FFTW_ESTIMATE) {
	return {in, out, FFTW_FORWARD, flags};
}

template<class In, class Out>
plan backward(const In &in, Out &out, unsigned int flags = FFTW_ESTIMATE) {
	return {in, out, FFTW_BACKWARD, flags};
}

};


#endif
