#ifndef __CHAMBOLLE_POCK_H__
#define __CHAMBOLLE_POCK_H__

#include "config.h"
#include "multi_array.h"
#include <vector>
#include <functional>
#include <string>
#include <memory>
#include <iostream>

#include <fstream>

#include <boost/filesystem.hpp>






// Implementation type.
enum impl_t {
	CPU_IMPL, GPU_IMPL
};


template<class T>
struct debug_state {
	boost::multi_array<T, 2> img;
	std::string name;
	debug_state(boost::multi_array<T, 2> img, std::string name = "") : img(img), name(name) {}
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

typedef std::array<size_t, 2> size2_t;

template<class T>
struct impl;

template<class T>
struct params {
	size_t max_steps = 10, monte_carlo_steps = 1000;
	T alpha, tau, sigma, input_variance = 1;
	bool debug = false, no_cache = false, penalized_scan = false;
	impl_t implementation;
	std::vector<size_t> kernel_sizes;
	size2_t size;

	params(size2_t size = {{0,0}}, std::vector<size_t> kernel_sizes = std::vector<size_t>(), T alpha = 0.5, T tau = 50, T sigma = 1)
	: alpha(alpha), tau(tau), sigma(sigma), implementation(CPU_IMPL), kernel_sizes(kernel_sizes), size(size) {}

	impl<T> *runner() const;

	template<class A>
	void set_size(A a) {
		size[0] = *a++;
		size[1] = *a++;
	}
};


template<class T>
struct impl {
	params<T> p;
	std::vector<debug_state<T>> debug_log;

	impl(const params<T> &p) : p(p), debug_log() {}

	virtual boost::multi_array<T, 2> run(const boost::multi_array<T,2> &) = 0;

protected:
	T cached_q(std::function<std::vector<T>()> calc) {
		static const auto cache_dir = "cache/";
		using namespace std;
		using namespace boost::filesystem;
		// filename from kernel stack
		ostringstream ss; ss << cache_dir << p.size[0] << 'x' << p.size[1];
		for(auto s : p.kernel_sizes)
			ss << '+' << s;
		auto target = ss.str();
		create_directories(cache_dir);
		vector<T> qs;
		if(p.no_cache || !exists(target)) {
			// simulate
			qs = calc();
			if(qs.size() == 0) return 4;
			sort(qs.begin(), qs.end());
			// write raw output.
			ofstream f(target);
			for(auto x : qs) f << x << '\n';
		} else {
			// read
			ifstream f(target);
			for(T value ; f >> value ; qs.push_back(value));
			sort(qs.begin(), qs.end());
		}
		// return (1 - alpha) quantile:
		return qs[size_t((qs.size() - 1) * (1 - p.alpha))];
	}

};

template<impl_t, class T>
struct chambolle_pock : impl<T> {};



#include "chambolle_pock_cpu.h"
#include "chambolle_pock_cl.h"

template<class T>
impl<T> *params<T>::runner() const {
	switch(implementation) {
		case CPU_IMPL: return new chambolle_pock<CPU_IMPL, T>(*this);
		case GPU_IMPL: return new chambolle_pock<GPU_IMPL, T>(*this);
		default: throw std::runtime_error("Unsupported implementation");
	}
}

#endif
