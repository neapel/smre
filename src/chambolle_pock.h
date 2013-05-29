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

#include "resolvent.h"
#include "constraint_parser.h"


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

template<class T>
struct impl;

template<class T>
struct params {
	size_t max_steps = 10, monte_carlo_steps = 1000;
	T alpha = 0.5, tau = 1000, sigma = 0.1, input_variance = -1, force_q = -1;
	bool no_cache = false, penalized_scan = false, dump_mc = false, use_fft = true, use_gpu = false;
	sizes_t kernel_sizes;
	size2_t size;
	std::shared_ptr<resolvent_params<T>> resolvent = std::make_shared<resolvent_l2_params<T>>();

	params(size2_t size = {{0,0}}, std::vector<size_t> kernel_sizes = std::vector<size_t>())
	: kernel_sizes(kernel_sizes), size(size) {}

	std::shared_ptr<impl<T>> runner() const;

	template<class A>
	void set_size(A a) {
		size[0] = *a++;
		size[1] = *a++;
	}
};


template<class T>
struct impl {
	const params<T> &p;
	T q, input_variance;

	// current progress [0:1]
	std::function<void(double, std::string desc)> progress_cb{nullptr};
	// current status. Breaks if true is returned, result is current status.
	std::function<bool(const boost::multi_array<T, 2> &, size_t step)> current_cb{nullptr};
	// debug.
	std::function<void(const boost::multi_array<T, 2> &, std::string desc)> debug_cb{nullptr};

	impl(const params<T> &p) : p(p) {}
	virtual ~impl() {}

	virtual boost::multi_array<T, 2> run(const boost::multi_array<T,2> &) = 0;

	virtual void progress(double q, std::string d) {
		if(progress_cb) progress_cb(q, d);
	}

	virtual bool current(const boost::multi_array<T,2> &a, size_t s) {
		if(current_cb) return current_cb(a, s);
		return true;
	}

	virtual void debug(const boost::multi_array<T,2> &a, std::string d) {
		if(debug_cb) debug_cb(a, d);
	}

protected:
	T cached_q(std::function<void(std::vector<std::vector<T>>&)> calc) {
		if(p.force_q >= 0) return p.force_q;

		static const auto cache_dir = "cache/";
		using namespace std;
		using namespace boost::filesystem;
		// filename from kernel stack
		ostringstream ss; ss << cache_dir << p.size[0] << 'x' << p.size[1];
		for(auto s : p.kernel_sizes)
			ss << '+' << s;
		if(p.penalized_scan) ss << "-penalized";
		auto target = ss.str();
		create_directories(cache_dir);
		vector<T> qs;
		if(p.no_cache || !exists(target)) {
			// simulate. k_qs[kernel][runs]
			const size_t N = p.kernel_sizes.size(), M = p.monte_carlo_steps;
			vector<vector<T>> k_qs;
			for(size_t i = 0 ; i < N ; i++) k_qs.emplace_back();
			calc(k_qs);
			// print raw data
			if(p.dump_mc) {
				ofstream o("mc.dat");
				for(size_t i = 0 ; i < N ; i++) {
					if(i != 0) o << '\t';
					o << p.kernel_sizes[i];
				}
				for(size_t j = 0 ; j < M ; j++)
					for(size_t i = 0 ; i < N ; i++)
						o << (i == 0 ? '\n' : '\t') << k_qs[i][j];
			}
			// max for each kernel. qs[runs]
			qs = k_qs[0];
			for(size_t i = 1 ; i < N ; i++)
				for(size_t j = 0 ; j < M ; j++)
					qs[j] = max(qs[j], k_qs[i][j]);
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

#include "chambolle_pock_cpu.h"
#include "chambolle_pock_cl.h"

template<class T>
std::shared_ptr<impl<T>> params<T>::runner() const {
	// verify
	if(kernel_sizes.size() < 1) throw std::invalid_argument("no kernels");
	if(resolvent == nullptr) throw std::invalid_argument("no resolvent");
	const auto min_sz = std::min(size[0], size[1]);
	for(auto k : kernel_sizes)
		if(k < 1 || k > min_sz) throw std::invalid_argument("invalid kernel size");
	// ok.
	if(use_gpu) return std::make_shared<chambolle_pock_gpu<T>>(*this);
	else return std::make_shared<chambolle_pock_cpu<T>>(*this);
}

#endif
