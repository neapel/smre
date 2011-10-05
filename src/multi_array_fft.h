#ifndef __MULTI_ARRAY_FFT_H__
#define __MULTI_ARRAY_FFT_H__

extern "C" {
	#include <fftw3.h>
}

#include <iostream>
#include <stdexcept>
#include <boost/multi_array.hpp>
#include <array>


template<class A>
class fft {
public:
	typedef A array_t;

private:
	static const int N = array_t::dimensionality;
	fftw_plan plan;


	void calculate_stride(const array_t &a, std::array<int, N> &nembed, int &stride) {
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

	bool is_continuous(const array_t &a) {
		typename array_t::size_type size = a.shape()[0];
		for(int i = 0 ; i < N ; i++)
			size *= a.strides()[i];
		return size == a.num_elements();
	}


	void make_plan(array_t &in, array_t &out, std::array<fftw_r2r_kind, N> kinds, unsigned flags) {
		std::array<typename array_t::index, N> in_base, out_base;
		std::copy(&in.index_bases()[0], &in.index_bases()[N], in_base.begin());
		std::copy(&out.index_bases()[0], &out.index_bases()[N], out_base.begin());

		double *in_data = &in(in_base);
		double *out_data = &out(out_base);

		std::array<int, N> shape;
		for(int i = 0 ; i < N ; i++)
			shape[i] = static_cast<int>(in.shape()[i]);

#if 0
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
	fft(array_t &in, array_t &out, fftw_r2r_kind kind, unsigned flags = 0) {
		std::array<fftw_r2r_kind, N> kinds;
		std::fill( kinds.begin(), kinds.end(), kind );
		make_plan(in, out, kinds, flags);
	}

	fft(array_t &in, array_t &out, std::array<fftw_r2r_kind, N> kinds, unsigned flags = 0) {
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



template<class A>
fft<A> plan_dct(A &in, A &out, unsigned flags = 0) {
	return fft<A>(in, out, FFTW_REDFT10, flags);
}

template<class A>
fft<A> plan_idct(A &in, A &out, unsigned flags = 0) {
	return fft<A>(in, out, FFTW_REDFT01, flags);
}


#endif
