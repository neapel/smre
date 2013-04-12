#ifndef __RESOLVENT_H__
#define __RESOLVENT_H__

#include <boost/multi_array.hpp>
#include <vector>
#include <iostream>
#include "multi_array.h"
#include "multi_array_fft.h"
#include <vexcl/vexcl.hpp>
#include "multi_array_operators.h"

template<impl_t I, class T>
struct resolvent_impl;

template<class T>
struct resolvent_impl<CPU_IMPL, T> {
	const T gamma;
	resolvent_impl(T gamma) : gamma(gamma) {}

	virtual void evaluate(T tau, const boost::multi_array<T, 2> &in, boost::multi_array<T, 2> &out) = 0;
};

template<class T>
struct resolvent_impl<GPU_IMPL, T> {
	const T gamma;
	resolvent_impl(T gamma) : gamma(gamma) {}

	virtual void evaluate(T tau, const vex::vector<T> &in, vex::vector<T> &out) = 0;
};

template<class T>
struct resolvent_params {
	// TODO: `template<impl_t> virtual` is forbidden...
	virtual resolvent_impl<CPU_IMPL, T> *cpu_runner(size2_t) const = 0;
	virtual resolvent_impl<GPU_IMPL, T> *gpu_runner(size2_t) const = 0;
};


/**
 * abstract resolvent class. It implements the l2 resolvent as default, i.e.
 * given \f$\tau>0\f$ and \f$ u \in \Omega = \mathbb R^I\f$ it evaluates the resolvent
 *
 * \f{\equation*}{
 * \left( \text{id} + \tau \partial J_\delta \right)^{-1}(u)
 * }
 *
 * with
 *
 * \f{\equation*}{
 * J_\delta(u) =  \frac{1}{2}\|u\|^2.
 * }
 *
 * This amount to the simple shrinkage \f$ u/(1+\tau) \f$. Classes derived from this basic
 * resolvent class, have to implement the evaluation of the resolvent.
 */
template<impl_t I, class T>
struct resolvent_l2 : public resolvent_impl<I, T> {};

template<class T>
struct resolvent_l2<CPU_IMPL, T> : public resolvent_impl<CPU_IMPL, T> {
	resolvent_l2() : resolvent_impl<CPU_IMPL, T>(1) {}

	virtual void evaluate(T tau, const boost::multi_array<T, 2> &in, boost::multi_array<T, 2> &out) {
		using namespace mimas;
		out = in;
		out /= 1 + tau;
	}
};

template<class T>
struct resolvent_l2<GPU_IMPL, T> : public resolvent_impl<GPU_IMPL, T> {
	resolvent_l2() : resolvent_impl<GPU_IMPL, T>(1) {}

	virtual void evaluate(T tau, const vex::vector<T> &in, vex::vector<T> &out) {
		out = in / (1 + tau);
	}
};

template<class T>
struct resolvent_l2_params : public resolvent_params<T> {
	virtual resolvent_impl<CPU_IMPL, T> *cpu_runner(size2_t) const {
		return new resolvent_l2<CPU_IMPL, T>();
	}
	virtual resolvent_impl<GPU_IMPL, T> *gpu_runner(size2_t) const {
		return new resolvent_l2<GPU_IMPL, T>();
	}
};

/**
 * given \f$\alpha>0\f$ and \f$ y \in \Omega = \mathbb R^I\f$ solve
 *
 * \f{align*}{
 *	\Delta u - \alpha u & = y \quad \text{ in } \Omega\\
 *	\langle \nabla u, \nu \rangle & = 0\quad\text{ on } \partial\Omega.
 * \f}
 *
 * The DCT I/III is used for fast implementation.
 */

template<class T>
boost::multi_array<T, 2> laplacian(size2_t size) {
	const size_t m = size[0], n = size[1];
	const std::array<size_t,1> me = {{m}}, ne = {{n}};
	fftw::plan<T, T, 1> dct_m(me), dct_n(ne);
	boost::multi_array<T, 1> eye1(me), dl1(me), de1(me), eye2(ne), dl2(ne), de2(ne);
	fill(eye1, 0.0);
	fill(eye2, 0.0);
	eye1[0] = eye2[0] = -1;
	eye1[1] = eye2[1] = 1;
	dct_m(eye1, dl1);
	dct_n(eye2, dl2);
	eye1[0] = eye2[0] = 1;
	eye1[1] = eye2[1] = 0;
	dct_m(eye1, de1);
	dct_n(eye2, de2);
	boost::multi_array<T, 2> out(size);
	for(size_t i = 0 ; i < m ; i++)
		for(size_t j = 0 ; j < n ; j++)
			out[i][j] = dl1[i] / de1[i] + dl2[j] / de2[j];
	return out;
}

template<impl_t, class T>
struct helmholtz {};

template<class T>
struct helmholtz<CPU_IMPL, T> {
	// discrete cosine transform of the laplacian
	boost::multi_array<T, 2> laplace_dct, temp;
	fftw::plan<T, T, 2> dct, idct;

	helmholtz(size2_t size)
	: laplace_dct(laplacian<T>(size)), temp(size),
	  dct(size, fftw::forward), idct(size, fftw::inverse) {}

	void solve(const T alpha, const boost::multi_array<T, 2> &in, boost::multi_array<T, 2> &out) {
		const size_t m = temp.shape()[0], n = temp.shape()[1];
		const T scale = 1.0 / (4 * m * n);
		dct(in, temp);
		for(size_t i = 0 ; i < m ; i++)
			for(size_t j = 0 ; j < n ; j++)
				temp[i][j] = scale * temp[i][j] / (laplace_dct[i][j] - alpha);
		idct(temp, out);
	}
};

template<class T>
struct helmholtz<GPU_IMPL, T> {
	// discrete cosine transform of the laplacian
	boost::multi_array<T, 2> laplace_dct, temp, temp2;
	fftw::plan<T, T, 2> dct, idct;

	helmholtz(size2_t size)
	: laplace_dct(laplacian<T>(size)), temp(size), temp2(size),
	  dct(size, fftw::forward), idct(size, fftw::inverse) {}

	void solve(const T alpha, const vex::vector<T> &in, vex::vector<T> &out) {
		// TODO: do this on GPU.
		const size_t m = temp.shape()[0], n = temp.shape()[1];
		const T scale = 1.0 / (4 * m * n);
		vex::copy(in, temp2.data());
		dct(temp2, temp);
		for(size_t i = 0 ; i < m ; i++)
			for(size_t j = 0 ; j < n ; j++)
				temp[i][j] = scale * temp[i][j] / (laplace_dct[i][j] - alpha);
		idct(temp, temp2);
		vex::copy(temp2.data(), out);
	}
};

/**
 * given \f$\tau>0\f$, \f$\delta\in [0,1]\f$ and \f$ u \in \Omega = \mathbb R^I\f$
 * evaluate the resolvent
 *
 * \f{\equation*}{
 * \left( \text{id} + \tau \partial J_\delta \right)^{-1}(u)
 * }
 *
 * with
 *
 * \f{\equation*}{
 * J_\delta(u) = \frac{\delta}{2} \| \nabla u\|_2^2 + \frac{1-\delta}{2}\|u\|^2.
 * }
 *
 * Setting \f$( \text{id} + \tau \partial J_\delta)^{-1}(u) = v$ shows that \f$u\f$ solves
 * the Helmholtz equation
 *
 * \f{align*}{
 *	\Delta u - \frac{1+t(1-\delta)}{\tau\delta} u & = \frac{-v}{\tau \delta} \quad \text{ in } \Omega\\
 *	\langle \nabla u, \nu \rangle & = 0\quad\text{ on } \partial\Omega.
 * \f}
 */
template<class T>
struct resolvent_h1_params : public resolvent_params<T> {
	const T delta;
	resolvent_h1_params(T delta = 0.5) : delta(delta) {}
	virtual resolvent_impl<CPU_IMPL, T> *cpu_runner(size2_t) const;
	virtual resolvent_impl<GPU_IMPL, T> *gpu_runner(size2_t) const;
};

template<impl_t I, class T>
struct resolvent_h1 : public resolvent_impl<I, T> {};

template<class T>
struct resolvent_h1<CPU_IMPL, T> : public resolvent_impl<CPU_IMPL, T> {
	const resolvent_h1_params<T> p;
	helmholtz<CPU_IMPL, T> h;
	resolvent_h1(resolvent_h1_params<T> p, size2_t size)
	: resolvent_impl<CPU_IMPL, T>(1 - p.delta), p(p), h(size) {}

	virtual void evaluate(T tau, const boost::multi_array<T, 2> &in, boost::multi_array<T, 2> &out) {
		using namespace mimas;
		const T alpha = (1 + tau * (1 - p.delta)) / (tau * p.delta);
		out = in;
		out *= 1 / (-tau * p.delta);
		h.solve(alpha, out, out);
	}
};

template<class T>
struct resolvent_h1<GPU_IMPL, T> : public resolvent_impl<GPU_IMPL, T> {
	const resolvent_h1_params<T> p;
	helmholtz<GPU_IMPL, T> h;
	resolvent_h1(resolvent_h1_params<T> p, size2_t size)
	: resolvent_impl<GPU_IMPL, T>(1 - p.delta), p(p), h(size) {}

	virtual void evaluate(T tau, const vex::vector<T> &in, vex::vector<T> &out) {
		const T alpha = (1 + tau * (1 - p.delta)) / (tau * p.delta);
		out = in / (-tau * p.delta);
		h.solve(alpha, out, out);
	}
};

template<class T>
resolvent_impl<CPU_IMPL, T> *resolvent_h1_params<T>::cpu_runner(size2_t size) const {
	return new resolvent_h1<CPU_IMPL, T>(*this, size);
}

template<class T>
resolvent_impl<GPU_IMPL, T> *resolvent_h1_params<T>::gpu_runner(size2_t size) const {
		return new resolvent_h1<GPU_IMPL, T>(*this, size);
}

#endif
