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

enum dir_t { forward, inverse };

template<class T> struct fftw_map {};
template<> struct fftw_map<float> { typedef float type; };
template<> struct fftw_map<std::complex<float>> { typedef fftwf_complex type; };

template<size_t dims>
struct size_a {
	std::array<int, dims> d, real;
	template<class I>
	size_a(I size) {
		for(size_t i = 0 ; i < dims ; i++) d[i] = size[i];
		for(size_t i = 0 ; i < dims - 1 ; i++) real[i] = size[i];
		real[dims - 1] = size[dims - 1] / 2 + 1;
	}
	operator const int*() {
		return d.data();
	}
};

// Data as fftw-compatible pointers.
template<class T, size_t dims>
typename fftw_map<T>::type *data(boost::multi_array<T, dims> &in) {
	return reinterpret_cast<typename fftw_map<T>::type *>(in.data());
}

template<class T, size_t dims>
typename fftw_map<T>::type *data(const boost::multi_array<T, dims> &in) {
	typedef typename fftw_map<T>::type U;
	return const_cast<U *>(reinterpret_cast<const U*>(in.data()));
}

#define common()\
	typedef boost::multi_array<T0, dims> A0; \
	typedef boost::multi_array<T1, dims> A1; \
	typedef plan<T0, T1, dims> this_t; \
	fftwf_plan p; \
	plan() : p(NULL) {} \
	plan(this_t &other) : p(other.p) { other.p = NULL; } \
	plan(this_t &&other) : p(other.p) { other.p = NULL; } \
	this_t &operator=(this_t &other) { p = other.p; other.p = NULL; return *this; } \
	this_t &operator=(this_t &&other) { p = other.p; other.p = NULL; return *this; } \
	~plan() { \
		if(p != NULL) \
			fftwf_destroy_plan(p); \
	}

template<class T0, class T1, size_t dims>
struct plan {};

template<size_t dims>
struct plan<float, float, dims> {
	typedef float T0;
	typedef float T1;
	common()

	template<class I>
	plan(I size, dir_t dir = forward, unsigned int flags = 0) {
		size_a<dims> sz(size);
		A0 in(size);
		A1 out(size);
		fftw_r2r_kind kind[dims];
		for(size_t i = 0 ; i < dims ; i++)
			kind[i] = dir == forward ? FFTW_REDFT10 : FFTW_REDFT01;
		#pragma omp critical
		p = fftwf_plan_r2r(dims, sz, data(in), data(out), kind, flags);
	}

	void operator()(const A0 &in, A1 &out) {
		fftwf_execute_r2r(p, data(in), data(out));
	}
};

template<size_t dims>
struct plan<float, std::complex<float>, dims> {
	typedef float T0;
	typedef std::complex<float> T1;
	common()

	template<class I>
	plan(I size, unsigned int flags = 0) {
		size_a<dims> sz(size);
		A0 in(sz.real);
		A1 out(size);
		#pragma omp critical
		p = fftwf_plan_dft_r2c(dims, sz, data(in), data(out), flags);
	}

	void operator()(const A0 &in, A1 &out) {
		fftwf_execute_dft_r2c(p, data(in), data(out));
	}
};

template<size_t dims>
struct plan<std::complex<float>, std::complex<float>, dims> {
	typedef std::complex<float> T0;
	typedef std::complex<float> T1;
	common()

	template<class I>
	plan(I size, dir_t dir = forward, unsigned int flags = 0) {
		size_a<dims> sz(size);
		A0 in(size);
		A1 out(size);
		p = fftwf_plan_dft(dims, sz, data(in), data(out),
			dir == forward ? FFTW_FORWARD : FFTW_BACKWARD, flags);
	}

	void operator()(const A0 &in, A1 &out) {
		fftwf_execute_dft(p, data(in), data(out));
	}
};

template<size_t dims>
struct plan<std::complex<float>, float, dims> {
	typedef std::complex<float> T0;
	typedef float T1;
	common()

	template<class I>
	plan(I size, unsigned int flags = 0) {
		size_a<dims> sz(size);
		A0 in(size);
		A1 out(sz.real);
		#pragma omp critical
		p = fftwf_plan_dft_c2r(dims, sz, data(in), data(out), flags);
	}

	void operator()(const A0 &in, A1 &out) {
		fftwf_execute_dft_c2r(p, data(in), data(out));
	}
};


}
#endif
