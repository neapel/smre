#ifndef __RESOLVENT_H__
#define __RESOLVENT_H__

#include <boost/multi_array.hpp>
#include <vector>
#include <iostream>
#include "multi_array.h"
#include "multi_array_fft.h"
#include <vexcl/vexcl.hpp>
#include "multi_array_operators.h"

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

template<class T>
struct resolvent {
	const T gamma;
	resolvent(T gamma = 1): gamma(gamma) {}

	virtual void evaluate(T tau, const boost::multi_array<T, 2> &in, boost::multi_array<T, 2> &out) {
		using namespace mimas;
		out = in;
		out /= 1 + tau;
	}

	virtual void evaluate(T tau, const vex::vector<T> &in, vex::vector<T> &out) {
		out = in / (1 + tau);
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
class helmholtz{

protected:
	// dimension
	size_t m,n;
	// discrete cosine transform of the laplacian
	boost::multi_array<T, 2> laplace_dct;
	fftw::plan<float, float, 2> dct, idct;

public:
	helmholtz(const size_t m, const size_t n)
	: m(m), n(n),
	  laplace_dct(boost::extents[m][n]),
	  dct(boost::extents[m][n], fftw::forward),
	  idct(boost::extents[m][n], fftw::inverse) {
		fftw::plan<float, float, 1> dct_m(boost::extents[m]), dct_n(boost::extents[n]);
		boost::multi_array<float, 1> eye1(boost::extents[m]), eye2(boost::extents[n]);
		boost::multi_array<float, 1> dl1(eye1), de1(eye1), dl2(eye2), de2(eye2);
		fill(eye1,0.0); fill(eye2,0.0);

		eye1[0] = -1.0, eye1[1] = 1.0, eye2[0] = -1.0, eye2[1] = 1.0;
		dct_m(eye1, dl1);
		dct_n(eye2, dl2);

		eye1[0] = 1.0, eye1[1] = 0.0, eye2[0] = 1.0, eye2[1] = 0.0;
		dct_m(eye1, de1);
		dct_n(eye2, de2);

		for(size_t i=0; i<m; i++)
			for(size_t j=0; j<n; j++)
				laplace_dct[i][j] = dl1[i]/de1[i] + dl2[j]/de2[j];
	};

	boost::multi_array<float, 2> solve(const boost::multi_array<float, 2> rhs, const float alpha){
		boost::multi_array<float, 2> rhs_dct(rhs), tmp(rhs), res(rhs);
		dct(rhs, rhs_dct);

		float m=rhs.shape()[1], n=rhs.shape()[2];
		float factor = 1/(4*m*n);

		for(size_t i=0; i<m; i++)
			for(size_t j=0; j<n; j++)
				tmp[i][j] = factor*rhs_dct[i][j] / (laplace_dct[i][j]-alpha);

		idct(tmp, res);
		return(res);

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
class h1resolvent : public helmholtz<T>, public resolvent<T> {
private:
	T delta;
public:
	h1resolvent(const int m, const int n, const T delta, const T tau) : helmholtz<T>(m,n), resolvent<T>(tau, 1 - delta), delta(delta) {}

	virtual void evaluate(T tau, const boost::multi_array<T, 2> &in, boost::multi_array<T, 2> &out) {
		T alpha = (1 + tau * (1 - delta)) / (tau * delta);
		out = in / (-tau * delta);
		solve(out, alpha);
	}

	virtual void evaluate(T tau, const vex::vector<T> &in, vex::vector<T> &out) {
		T alpha = (1 + tau * (1 - delta)) / (tau * delta);
		out = in / (-tau * delta);
		solve(out, alpha);
	}
};

#endif
