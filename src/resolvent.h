#ifndef __RESOLVENT_H__
#define __RESOLVENT_H__

#include <boost/multi_array.hpp>
#include <vector>
#include <iostream>
#include "multi_array.h"
#include "multi_array_fft.h"

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

class resolvent{
protected:
	// resolvent parameter
	float tau;
public:
	resolvent(): tau(1) {}
	resolvent(const float tau): tau(tau) {}

	inline void update_param(const float new_tau){ tau = new_tau; }
	virtual boost::multi_array<float, 2> evaluate(const boost::multi_array<float, 2> arg){
		boost::multi_array<float, 2> out(arg);
		const auto size = boost::extents_of(arg);
		const auto width = size[0], height = size[1];
		for(size_t ix = 0 ; ix <width ; ix++)
			for(size_t iy = 0 ; iy < height ; iy++)
				out[ix][iy] = arg[ix][iy]/(1+tau);

		return(out);
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

class helmholtz{

protected:
	// dimension
	size_t m,n;
	// discrete cosine transform of the laplacian
	boost::multi_array<float, 2> laplace_dct;

public:
	helmholtz(): m(1), n(1), laplace_dct(boost::extents[1][1]) {}
	helmholtz(const int m, const int n) : m(m), n(n), laplace_dct(boost::extents[m][n]) {
		boost::multi_array<float, 1> eye1(boost::extents[m]), eye2(boost::extents[n]);
		boost::multi_array<float, 1> dl1(eye1), de1(eye1), dl2(eye2), de2(eye2);
		fill(eye1,0.0); fill(eye2,0.0);

		eye1[0] = -1.0, eye1[1] = 1.0, eye2[0] = -1.0, eye2[1] = 1.0;
		fftw::forward(eye1,dl1)();
		fftw::forward(eye2,dl2)();

		eye1[0] = 1.0, eye1[1] = 0.0, eye2[0] = 1.0, eye2[1] = 0.0;
		fftw::forward(eye1,de1)();
		fftw::forward(eye2,de2)();


		for(int i=0; i<m; i++)
			for(int j=0; j<n; j++)
				laplace_dct[i][j] = dl1[i]/de1[i] + dl2[j]/de2[j];

	};

	//	void set_rhs(boost::multi_array<float, 2> & new_rhs){
	//		for(size_t i = 0 ; i < 2 ; i++)
	//			if(rhs.shape()[i] != new_rhs.shape()[i])
	//				throw std::invalid_argument("Input and output arrays must be of same size.");
	//		rhs = new_rhs;
	//	};
	//	void set_alpha(float new_alpha){alpha = new_alpha;};

	boost::multi_array<float, 2> solve(const boost::multi_array<float, 2> rhs, const float alpha){
		boost::multi_array<float, 2> rhs_dct(rhs), tmp(rhs), res(rhs);
		fftw::forward(rhs, rhs_dct)();

		float m=rhs.shape()[1], n=rhs.shape()[2];
		float factor = 1/(4*m*n);

		for(size_t i=0; i<m; i++)
			for(size_t j=0; j<n; j++)
				tmp[i][j] = factor*rhs_dct[i][j] / (laplace_dct[i][j]-alpha);

		fftw::backward(tmp, res)();
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

class h1resolvent : public helmholtz, public resolvent{
private:
	float delta;
public:
	h1resolvent() : helmholtz(), resolvent(),  delta(0) {}
	h1resolvent(const int m, const int n, const float delta, const float tau) : helmholtz(m,n), resolvent(tau), delta(delta) {}

	boost::multi_array<float, 2> evaluate(const boost::multi_array<float, 2> arg){
		float alpha = (1+tau*(1-delta))/(tau*delta);
		boost::multi_array<float, 2> rhs(arg);
		for(size_t ix = 0 ; ix < m ; ix++)
			for(size_t iy = 0 ; iy < n ; iy++)
				rhs[ix][iy] = -arg[ix][iy]/(tau*delta);
		return(solve(rhs, alpha));
	}
};

#endif
