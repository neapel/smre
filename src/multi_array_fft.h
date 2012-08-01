#ifndef __MULTI_ARRAY_FFT_H__
#define __MULTI_ARRAY_FFT_H__

extern "C" {
	#include <fftw3.h>
}
#include <boost/multi_array.hpp>
#include <array>
#include <stdexcept>
#include <complex>

#if 0
#include <iostream>
#include "../test/multi_array_compare.h"
template<class In, class Out>
class fft {
	static_assert(In::dimensionality == Out::dimensionality, "Arrays must have same rank.");
	static const size_t N = In::dimensionality;
	fftw_plan plan;

	template<class A>
	void calculate_stride(const A &a, std::array<int, N> &nembed, int &stride) {
		// FFTw calculates stride[i] = stride * nembed[i + 1] * ... * nembed[N - 1]
		// see: fftw-source:/api/mktensor-rowmajor.c

		stride = a.strides()[N - 1]; // innermost
		for(int i = 1 ; i < N ; i++) {
			const int l = a.strides()[i - 1];
			const int r = a.strides()[i];
			if( l % r != 0 )
				throw std::invalid_argument("Unsupported array stride, copy it into a new dense array.");
			nembed[i] = l / r;
		}
		// nembed[0] is unused?!
		nembed[0] = 0;
	}

	void make_plan(In &in, Out &out, std::array<fftw_r2r_kind, N> kinds, unsigned flags) {
#if 0
		std::array<int, N> in_base, out_base;
		std::copy(in.index_bases(), in.index_bases() + N, in_base.begin());
		std::copy(out.index_bases(), out.index_bases() + N, out_base.begin());

		// doesn't work with views?!
		double *in_data = &in(in_base);
		double *out_data = &out(out_base);
#else
		for(size_t i = 0 ; i < N ; i++)
			if(in.index_bases()[i] != 0 || out.index_bases()[i] != 0)
				throw std::invalid_argument("Nonzero bases are not supported");
		double *in_data = in.origin();
		double *out_data = out.origin();
#endif

		std::array<int, N> shape;
		for(size_t i = 0 ; i < N ; i++)
			shape[i] = static_cast<int>(in.shape()[i]);

#if 1
		if( !is_continuous(in) || !is_continuous(out) )
			throw std::invalid_argument("Views are not supported.");

		plan = fftw_plan_r2r(
			/* rank */ N,
			/* n */ shape.data(),
			/* in */ in_data,
			/* out */ out_data,
			kinds.data(),
			flags
		);
#else
		// calculate strides
		std::array<int, N> in_nembed, out_nembed;
		int in_stride, out_stride;
		calculate_stride(in, in_nembed, in_stride);
		calculate_stride(out, out_nembed, out_stride);

		plan = fftw_plan_many_r2r(
			/* rank */ N,
			/* n[N] */ shape.data(),
			/* howmany */ 1,
			/* in */ in_data,
			/* inembed[N] */ in_nembed.data(),
			/* istride */ in_stride,
			/* idist */ 0,
			/* out */ out_data,
			/* onembed[N] */ out_nembed.data(),
			/* ostride */ out_stride,
			/* odist */ 0,
			/* kind[N] */ kinds.data(),
			/* flags */ flags
		);
#endif
	}

public:
	fft(In &in, Out &out, fftw_r2r_kind kind, unsigned flags = 0) {
		std::array<fftw_r2r_kind, N> kinds;
		std::fill( kinds.begin(), kinds.end(), kind );
		make_plan(in, out, kinds, flags);
	}

	fft(In &in, Out &out, std::array<fftw_r2r_kind, N> kinds, unsigned flags = 0) {
		make_plan(in, out, kinds, flags);
	}

	// don't want to track plan ownership.
	fft() = delete;
	fft(const fft &) = delete;
	fft(const fft && o) : plan(o.plan) {}


	~fft() {
		fftw_destroy_plan(plan);
	}

	
	/*! Execute this plan */
	void operator()() {
		fftw_execute(plan);
	}


};



template<class In, class Out>
fft<In, Out> plan_dct(In &in, Out &out, unsigned flags = 0) {
	return {in, out, FFTW_REDFT10, flags};
}

template<class In, class Out>
fft<In, Out> plan_idct(In &in, Out &out, unsigned flags = 0) {
	return {in, out, FFTW_REDFT01, flags};
}

#endif

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

template<class In, class Out>
struct plan {
};

/** Complex to Complex FFT. */
template<size_t dims>
struct plan<boost::multi_array<std::complex<float>, dims>, boost::multi_array<std::complex<float>, dims>> {
	typedef boost::multi_array<std::complex<float>, dims> in_t;
	typedef boost::multi_array<std::complex<float>, dims> out_t;
	fftwf_plan p;

	plan(const in_t &in, out_t &out, int dir, unsigned int flags) {
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

	void operator()(){
		fftwf_execute(p);	
	}

	~plan() {
		fftwf_destroy_plan(p);
	}
};


/** Real to Complex FFT. */
template<size_t dims>
struct plan<boost::multi_array<float, dims>, boost::multi_array<std::complex<float>, dims>> {
	typedef boost::multi_array<float, dims> in_t;
	typedef boost::multi_array<std::complex<float>, dims> out_t;
	fftwf_plan p;

	plan(const in_t &in, out_t &out, int dir, unsigned int flags) {
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

	void operator()(){
		fftwf_execute(p);	
	}

	~plan() {
		fftwf_destroy_plan(p);
	}
};


/** Complex to Real FFT. */
template<size_t dims>
struct plan<boost::multi_array<std::complex<float>, dims>, boost::multi_array<float, dims>> {
	typedef boost::multi_array<std::complex<float>, dims> in_t;
	typedef boost::multi_array<float, dims> out_t;
	fftwf_plan p;

	plan(const in_t &in, out_t &out, int dir, unsigned int flags) {
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

	void operator()(){
		fftwf_execute(p);	
	}

	~plan() {
		fftwf_destroy_plan(p);
	}
};

/** Todo: R2R, double */


template<class In, class Out>
plan<In, Out> forward(const In &in, Out &out, unsigned int flags = FFTW_ESTIMATE) {
	return {in, out, FFTW_FORWARD, flags};
}

template<class In, class Out>
plan<In, Out> backward(const In &in, Out &out, unsigned int flags = FFTW_ESTIMATE) {
	return {in, out, FFTW_BACKWARD, flags};
}

};


#endif
