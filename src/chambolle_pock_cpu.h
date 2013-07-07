#ifndef __CHAMBOLLE_POCK_CPU_H__
#define __CHAMBOLLE_POCK_CPU_H__

#include <boost/format.hpp>

#include "chambolle_pock.h"
#include "resolvent.h"
#include "convolution.h"
#include "image_variance.h"


#if HAVE_OPENMP
#include <omp.h>
#endif



template<class T>
struct chambolle_pock_cpu : public impl<T> {
	typedef boost::multi_array<T, 2> A;

	using impl<T>::p;
	using impl<T>::input_stddev;
	using impl<T>::q;

	struct constraint {
		// size of the box kernel.
		size_t k_size;
		std::shared_ptr<prepared_kernel> k, adj_k;
		// y for this constraint
		A y;
		// specific q for this constraint.
		T q, shift_q;

		constraint(size_t k_size, size2_t size,
			std::shared_ptr<prepared_kernel> k, std::shared_ptr<prepared_kernel> adj_k)
		: k_size(k_size), k(k), adj_k(adj_k), y(size), q(-1), shift_q(0) {}
	};

	inline T soft_shrink(T x, T q) {
		if(x < -q) return x + q;
		if(x > q) return x - q;
		return 0;
	}

	T total_norm;
	std::vector<constraint> constraints;
	std::shared_ptr<resolvent_cpu<T>> resolv;
	std::shared_ptr<cpu_convolver<T>> convolution;
	bool initialized = false;

	chambolle_pock_cpu(const params<T> &p)
	: impl<T>(p),
	  resolv(p.resolvent->cpu_runner(p.size)) {
		if(p.use_fft) convolution = std::make_shared<cpu_fft_convolver<T>>(p.size);
		else convolution = std::make_shared<cpu_sat_convolver<T>>(p.size);
#if HAVE_OPENMP
		if(this->profiler)
			omp_set_num_threads(1);
#endif
	}

	void update_kernels() {
		constraints.clear();
		total_norm = 0;
		for(auto k_size : p.kernel_sizes) {
			auto prep_k = convolution->prepare_kernel(k_size, false);
			auto adj_prep_k = convolution->prepare_kernel(k_size, true);
			constraints.emplace_back(k_size, p.size, prep_k, adj_prep_k);
			total_norm += k_size * k_size / 2;
		}
		if(p.penalized_scan)
			for(auto &c : constraints)
				c.shift_q = sqrt(log(1.0 * p.size[0] * p.size[1] / pow(c.k_size, 2)));
		calc_q();
	}

	void calc_q() {
		using namespace mimas;
		using namespace std;
		q = this->cached_q([&](std::vector<std::vector<T>> &k_qs){
			random_device dev;
			const size_t seed = dev();
			#pragma omp parallel for
			for(size_t i = 0 ; i < p.monte_carlo_steps ; i++) {
				A data(p.size), convolved(p.size);
				mt19937 gen(seed + i);
				normal_distribution<T> dist(/*mean*/0, /*stddev*/1);
				for(auto row : data) for(auto &x : row) x = dist(gen);
				auto f_data = convolution->prepare_image(data);
				for(size_t j = 0 ; j < constraints.size() ; j++) {
					convolution->conv(f_data, constraints[j].k, convolved);
					auto k_q = norm_inf(convolved) - constraints[j].shift_q;
					#pragma omp critical
					k_qs[j].push_back(k_q);
				}
#if HAVE_OPENMP
				if(omp_get_thread_num() == 0)
					this->progress(double(i * omp_get_num_threads()) / p.monte_carlo_steps, "Monte Carlo simulation for q");
#else
				this->progress(double(i) / p.monte_carlo_steps, "Monte Carlo simulation for q");
#endif
			}
		});
		for(auto &c : constraints)
			c.q = q + c.shift_q;
	}


	bool current(const A &a, size_t s) {
		if(this->current_cb) {
			return this->current_cb(a, s);
		}
		return true;
	}

	void debug(const A &a, std::string d) {
		if(this->debug_cb) {
			#pragma omp critical
			this->debug_cb(a, d);	
		}
	}

	void profile_push(std::string name) {
		if(this->profiler)
			this->profiler->tic_cpu(name);
	}

	void profile_pop() {
		if(this->profiler)
			this->profiler->toc("");
	}

	virtual A run(const A &Y_) {
		using namespace mimas;

		auto Y = Y_;
#ifdef DEBUG_WATERMARK
		for(size_t i0 = 0 ; i0 < 20 ; i0++)
			for(size_t i1 = 0 ; i1 < 20 ; i1++)
				Y[i0 + 20][i1 + p.size[1] - 40] = 0;		
#endif

		profile_push("run");
		profile_push("allocate");
			A x(Y), bar_x(Y), old_x(p.size), w(p.size), out(p.size);
		profile_pop();

		if(!initialized) {
			profile_push("update kernels");
				update_kernels();
			profile_pop();
			initialized = true;
		}

		if(p.input_stddev >= 0)
			input_stddev = p.input_stddev;
		else
			input_stddev = median_absolute_deviation(Y_);

		this->debug(x, "x_in");
		for(auto &c : constraints) {
			fill(c.y, 0);
		}

		T tau = p.tau;
		T sigma = p.sigma;

		// Adjust sigma with norm.
		sigma /= tau * total_norm;

		// Repeat until good enough.
		profile_push("iteration");
		for(size_t n = 0 ; n < p.max_steps ; n++) {
			profile_push("step");
			// reset accumulator
			profile_push("(a) reset w");
				fill(w, 0);
			profile_pop();
			// transform bar_x for convolutions
			profile_push("(b) prepare bar_x");
				const auto f_bar_x = convolution->prepare_image(bar_x);
			profile_pop();
			profile_push("constraints");
			#pragma omp parallel for
			for(size_t i = 0 ; i < constraints.size() ; i++) {
				profile_push("kernel");
				auto &c = constraints[i];
				// convolve bar_x with kernel
				profile_push("(c) k * bar_x");
					A convolved(p.size);
					convolution->conv(f_bar_x, c.k, convolved);
				profile_pop();
				debug(convolved, str(boost::format("convolved_%d") % i));
				// calculate new y_i
				profile_push("(d) soft_shrink");
					convolved *= sigma;
					c.y += convolved;
					for(auto row : c.y) for(auto &v : row) v = soft_shrink(v, c.q * sigma * input_stddev);
				profile_pop();
				debug(c.y, str(boost::format("y_%d") % i));
				// convolve y_i with conjugate transpose of kernel
				profile_push("(e) prepare y");
					const auto f_y = convolution->prepare_image(c.y);
				profile_pop();
				profile_push("(f) adj_k * y");
					convolution->conv(f_y, c.adj_k, convolved);
				profile_pop();
				debug(convolved, str(boost::format("adj_convolved_%d") % i));
				// accumulate
				profile_push("(g) accumulate w");
					#pragma omp critical
					w += convolved;
				profile_pop();
				profile_pop(/*kernel*/);
				debug(w, str(boost::format("w_%d") % i));
			}
			profile_pop();
			this->progress(double(n) / p.max_steps, str(boost::format("Chambolle-Pock step %d") % n));

			profile_push("(h) resolvent");
				old_x = x;
				w *= tau; bar_x = x; bar_x -= Y; bar_x -= w;
				debug(bar_x, "resolv_in");
				resolv->evaluate(tau, bar_x, x);
				x += Y;
			profile_pop();
			debug(x, "resolv_out");
			const T theta = 1 / sqrt(1 + 2 * tau * resolv->gamma);
			tau *= theta;
			sigma /= theta;
			profile_push("(i) bar_x");
				bar_x = x; bar_x -= old_x; bar_x *= theta; bar_x += x;
				debug(bar_x, "bar_x");

				out = Y; out -= x;
			profile_pop();
			profile_pop(/*step*/);

			if(!current(out, n)) break;

			if(n > 1 && p.tolerance > 0) {
				const T ch = norm_1(x) / norm_1(x - old_x);
				if(ch >= p.tolerance) break;
			}
		}
		profile_pop();
		profile_pop(/*run*/);
		return out;
	}
};
#endif




