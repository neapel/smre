#ifndef __CHAMBOLLE_POCK_H__
#define __CHAMBOLLE_POCK_H__

#include "multi_array.h"
#include <vector>
#include <functional>
#include <string>
#include <memory>

/**
 * One constraint: \f$ a \le (k * x)_w \le b \quad\forall w \in I \f$
 */
struct constraint {
	// clamp
	float a, b;
	constraint(float a, float b) : a(a), b(b) {}

	// lazy generator
	virtual boost::multi_array<float, 2> get_k(const boost::multi_array<float, 2> &) = 0;
};

struct debug_state {
	const boost::multi_array<float, 2> &img;
	std::string name;
	int n, i;
	float tau, sigma, theta;
};


/**
 * given \f$\tau_0, \sigma_0, K_i, x^0, y^0_i \in \mathbb R^I\f$.
 *
 * with \f$\sigma_0 > 0, \tau_0 \sigma_0 L^2 \le 1, \bar x^0 = x^0\f$ 
 *
 * for \f$n = 1,\dotsc\f$
 *
 * 	\f{align*}{
 * 		i = 1,\dotsc,N \qquad
 * 		y^{n + 1}_i &= (y^n_i + \sigma_n K_i * \bar x^n)
 * 			- \sigma_n \textrm{clamp}_{[a_i, b_i]}(y_i^n / \sigma_n + K_i * \bar x^n) \\
 * 		w_i &= K_i^* * y^{n + 1}_i \\
 * 		w &= \sum_i w_i \\
 * 		x^{n + 1} &= x^n - \tau_n w + \tau_n \gamma / (1 + \tau_n) \\
 * 		\theta_n &= 1 / \sqrt{1 + 2 \tau_n \gamma} \\
 * 		\tau_{n + 1} &= \theta_n \tau_n \\
 * 		\sigma_{n + 1} &= \sigma_n / \theta_n \\
 * 		\bar x^{n + 1} &= x^{n + 1} + \theta_n (x^{n + 1} - x^n)
 * 	\f}
 */
struct chambolle_pock {
	size_t max_steps;
	float tau, sigma, gamma;
	std::vector<std::shared_ptr<constraint>> constraints;
	std::vector<debug_state> debug_log;
	bool debug;

	chambolle_pock()
	: max_steps(10), tau(50), sigma(1), gamma(1) {}

	boost::multi_array<float, 2> run(const boost::multi_array<float, 2> &);

#if HAVE_OPENCL
	boost::multi_array<float, 2> run_cl(const boost::multi_array<float, 2> &);
#endif
};





#endif
